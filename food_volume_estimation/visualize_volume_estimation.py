
import sys
import os
import json
import matplotlib.pyplot as plt
# Add parent directory to sys.path to allow importing the package
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from tensorflow.keras.models import Model, model_from_json
from food_volume_estimation.volume_estimator import VolumeEstimator
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from pyntcloud import PyntCloud

# Paths to model architecture/weights (Updated paths)
current_dir = os.path.dirname(os.path.abspath(__file__))
depth_model_architecture = os.path.join(current_dir, '../models/depth_architecture.json')
depth_model_weights = os.path.join(current_dir, '../models/depth_weights.h5')
segmentation_model_weights = os.path.join(current_dir, '../models/best.onnx')

print("[*] Initializing VolumeEstimator...")

# Create estimator object and intialize
estimator = VolumeEstimator(arg_init=False)
with open(depth_model_architecture, 'r') as read_file:
    custom_losses = Losses()
    objs = {'ProjectionLayer': ProjectionLayer,
            'ReflectionPadding2D': ReflectionPadding2D,
            'InverseDepthNormalization': InverseDepthNormalization,
            'AugmentationLayer': AugmentationLayer,
            'compute_source_loss': custom_losses.compute_source_loss}
    model_architecture_json = json.load(read_file)
    estimator.monovideo = model_from_json(model_architecture_json, custom_objects=objs)

estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo, False)
estimator.monovideo.load_weights(depth_model_weights)
estimator.model_input_shape = estimator.monovideo.inputs[0].shape.as_list()[1:]
depth_net = estimator.monovideo.get_layer('depth_net')
estimator.depth_model = Model(inputs=depth_net.inputs, outputs=depth_net.outputs, name='depth_model')
print('[*] Loaded depth estimation model.')

# Depth model configuration
MIN_DEPTH = 0.01
MAX_DEPTH = 10
estimator.min_disp = 1 / MAX_DEPTH
estimator.max_disp = 1 / MIN_DEPTH
estimator.gt_depth_scale = 0.35 # Ground truth expected median depth

# Create segmentator object
estimator.segmentator = FoodSegmentator(segmentation_model_weights)

# Set plate adjustment relaxation parameter
estimator.relax_param = 0.01

print("[*] Estimating volume...")
# Estimate volumes in input image
input_image = os.path.join(current_dir, '../assets/readme_assets/examples/rice_example.jpg')
plate_diameter = 0.35 # Set as 0 to ignore plate detection and scaling

# For script usage, we set plot_results=True (user can see window or save file)
# We also provide a plots_directory to save the output files.
outputs_list = estimator.estimate_volume(input_image, fov=70, plate_diameter_prior=plate_diameter, 
                                         plot_results=True, plots_directory='output_plots')

print(f"[*] Got {len(outputs_list)} outputs.")

# Plot results for all detected food objects (Testing logic only)
for i, outputs in enumerate(outputs_list):
    (estimated_volume, object_points_df, non_object_points_df, plane_points_df, object_points_transformed_df, 
        plane_points_transformed_df, simplices) = outputs
    
    print(f"[*] Object {i}: Volume = {estimated_volume * 1000:.4f} L")
    
    # Flip x and z coordinates to match point cloud with plotting axes
    object_points_df.values[:,0] *= -1
    object_points_df.values[:,2] *= -1
    non_object_points_df.values[:,0] *= -1
    non_object_points_df.values[:,2] *= -1
    plane_points_df.values[:,0] *= -1
    plane_points_df.values[:,2] *= -1
    
print("[*] Estimation complete. Check 'output_plots' directory for results.")
