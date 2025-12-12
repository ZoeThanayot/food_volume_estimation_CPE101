import argparse
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model, model_from_json
from food_volume_estimation.volume_estimator import VolumeEstimator, DensityDatabase
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from flask import Flask, request, jsonify, make_response, abort
import base64
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io


app = Flask(__name__)
estimator = None
density_db = None

def load_volume_estimator(depth_model_architecture, depth_model_weights,
        segmentation_model_weights, density_db_source, segmentation_model_config=None):
    """Loads volume estimator object and sets up its parameters."""
    # Create estimator object and intialize
    global estimator
    estimator = VolumeEstimator(arg_init=False)
    with open(depth_model_architecture, 'r') as read_file:
        custom_losses = Losses()
        objs = {'ProjectionLayer': ProjectionLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer,
                'compute_source_loss': custom_losses.compute_source_loss}
        model_architecture_json = json.load(read_file)
        estimator.monovideo = model_from_json(model_architecture_json,
                                              custom_objects=objs)
    estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo,
                                                      False)
    estimator.monovideo.load_weights(depth_model_weights, by_name=True)
    estimator.model_input_shape = (
        estimator.monovideo.inputs[0].shape.as_list()[1:])
    depth_net = estimator.monovideo.get_layer('depth_net')
    estimator.depth_model = Model(inputs=depth_net.inputs,
                                  outputs=depth_net.outputs,
                                  name='depth_model')
    print('[*] Loaded depth estimation model.')

    # Depth model configuration
    MIN_DEPTH = 0.01
    MAX_DEPTH = 10
    estimator.min_disp = 1 / MAX_DEPTH
    estimator.max_disp = 1 / MIN_DEPTH
    estimator.gt_depth_scale = 0.35 # Ground truth expected median depth

    # Create segmentator object
    estimator.segmentator = FoodSegmentator(segmentation_model_weights, segmentation_model_config)
    # Set plate adjustment relaxation parameter
    estimator.relax_param = 0.01

    # Need to define default graph due to Flask multiprocessing
    global graph
    graph = tf.get_default_graph()
    global sess
    sess = tf.compat.v1.keras.backend.get_session()

    # Load food density database
    global density_db
    density_db = DensityDatabase(density_db_source)

# Classification Globals
classifier_model = None
classifier_labels = None
device = torch.device("cpu") # Use CPU to avoid conflict with TF if GPU is present

def load_classifier(model_path, label_map_path):
    global classifier_model, classifier_labels
    
    # Load Label Map
    with open(label_map_path, 'r', encoding='utf-8') as f:
        classifier_labels = json.load(f)
        
    # Load Model
    print('[*] Loading classification model...')
    classifier_model = models.efficientnet_b0(num_classes=100)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    classifier_model.load_state_dict(state_dict)
    
    classifier_model.to(device)
    classifier_model.eval()
    print('[*] Classification model loaded.')

@app.route('/classify', methods=['POST'])
def classify_food():
    """Classify the food in the image."""
    try:
        content = request.get_json()
        img_encoded = content['img']
        # Check if list or base64 string
        if isinstance(img_encoded, list):
             img_byte_string = ' '.join([str(x) for x in img_encoded])
             np_img = np.fromstring(img_byte_string, np.uint8, sep=' ')
             img_cv2 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        else:
             # Assume base64 or other handling if needed, but client sends list
             img_byte_string = ' '.join([str(x) for x in img_encoded])
             np_img = np.fromstring(img_byte_string, np.uint8, sep=' ')
             img_cv2 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            abort(406)

        # Convert to PIL Image for PyTorch transforms
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = classifier_model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        class_id = str(top_catid[0].item())
        confidence = top_prob[0].item()
        food_name = classifier_labels.get(class_id, "Unknown")
        
        return jsonify({
            "food_type": food_name,
            "confidence": round(confidence, 4),
            "class_id": class_id
        })
        
    except Exception as e:
        print(f"Error in classification: {e}")
        abort(500)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "running",
        "message": "Food Volume Estimation API is ready. Use /predict endpoint to estimate volume."
    })

@app.route('/predict', methods=['POST'])
def volume_estimation():
    """Receives an HTTP multipart request and returns the estimated 
    volumes of the foods in the image given.

    Multipart form data:
        img: The image file to estimate the volume in.
        plate_diameter: The expected plate diamater to use for depth scaling.
        If omitted then no plate scaling is applied.

    Returns:
        The array of estimated volumes in JSON format.
    """
    # Decode incoming byte stream to get an image
    try:
        content = request.get_json()
        img_encoded = content['img']
        img_byte_string = ' '.join([str(x) for x in img_encoded]) # If in byteArray
        #img_byte_string = base64.b64decode(img_encoded) # Decode if in base64
        np_img = np.fromstring(img_byte_string, np.uint8, sep=' ')
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        abort(406)



    # Get expected plate diameter from form data or set to 0 and ignore
    try:
        plate_diameter = float(content['plate_diameter'])
    except Exception as e:
        plate_diameter = 0

    # Estimate volumes
    # Estimate volumes
    with graph.as_default():
        with sess.as_default():
            volumes = estimator.estimate_volume(img, fov=70,
                plate_diameter_prior=plate_diameter)
    # Convert to mL
    # volumes is now a list of (volume_in_m3, class_name)
    
    results = []
    total_weight = 0
    total_nutrition = {'kcal': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'sugar': 0}
    
    for vol_m3, class_name in volumes:
        vol_ml = vol_m3 * 1e6
        
        # Query density DB with the detected class name
        # Expected return: [Food, Density, kcal, protein, carbs, fat, sugar]
        db_entry = density_db.query(class_name)
        
        # Handle cases where db_entry might not have all columns if fallback occurred
        # Default safety
        if len(db_entry) < 7:
             # Fallback if DB update didn't propagate or error return
             density = db_entry[1]
             nutrition_per_100g = {'kcal': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'sugar': 0}
        else:
            density = db_entry[1]
            nutrition_per_100g = {
                'kcal': db_entry[2],
                'protein': db_entry[3],
                'carbs': db_entry[4],
                'fat': db_entry[5],
                'sugar': db_entry[6]
            }
        
        item_weight_g = vol_ml * density
        total_weight += item_weight_g
        
        # Calculate item nutrition (Values are per 100g)
        item_nutrition = {}
        for key, val in nutrition_per_100g.items():
            # (Weight / 100) * Value
            n_val = (item_weight_g / 100.0) * float(val)
            item_nutrition[key] = round(n_val, 2)
            total_nutrition[key] += n_val
        
        item_result = {
            'food_type': class_name,
            'weight_g': round(float(item_weight_g), 2)
        }
        item_result.update(item_nutrition)
        results.append(item_result)

    # Round totals
    total_nutrition_rounded = {k: round(v, 2) for k, v in total_nutrition.items()}

    # Return values
    return_vals = {
        'total_weight_g': round(float(total_weight), 2),
        'total_nutrition': total_nutrition_rounded,
        'items': results
    }
    return jsonify(return_vals)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Food volume estimation API.')
    parser.add_argument('--depth_model_architecture', type=str,
                        help='Path to depth model architecture (.json).',
                        metavar='/path/to/architecture.json',
                        required=True)
    parser.add_argument('--depth_model_weights', type=str,
                        help='Path to depth model weights (.h5).',
                        metavar='/path/to/depth/weights.h5',
                        required=True)
    parser.add_argument('--segmentation_model_weights', type=str,
                        help='Path to segmentation model weights (.onnx).',
                        metavar='/path/to/weights.onnx',
                        required=True)
    parser.add_argument('--segmentation_model_config', type=str,
                        help='Path to segmentation model config (.yaml).',
                        metavar='/path/to/FoodSeg.yaml',
                        required=True)
    parser.add_argument('--density_db_source', type=str,
                        help=('Path to food density database (.xlsx) ' +
                              'or Google Sheets ID.'),
                        metavar='/path/to/plot/database.xlsx or <ID>',
                        required=True)
    parser.add_argument('--classifier_model', type=str,
                        help='Path to classifier model weights (.pth).',
                        metavar='/path/to/EfficientNet-B0.pth',
                        required=False)
    parser.add_argument('--classifier_label_map', type=str,
                        help='Path to classifier label map (.json).',
                        metavar='/path/to/Efficient_Label_Map.json',
                        required=False)
    args = parser.parse_args()

    load_volume_estimator(args.depth_model_architecture,
                          args.depth_model_weights, 
                          args.segmentation_model_weights,
                          args.density_db_source,
                          args.segmentation_model_config)
                          
    if args.classifier_model and args.classifier_label_map:
        load_classifier(args.classifier_model, args.classifier_label_map)
    else:
        print("[*] Classifier args not provided. /classify endpoint will error.")

    app.run(host='0.0.0.0', port=5001)
