import sys
import os
import cv2
import numpy as np
from food_volume_estimation.food_segmentation.yolo_onnx import YOLOSegmentation

class FoodSegmentator():
    """Food segmentator object using the YOLO ONNX model."""
    def __init__(self, weights_path, config_path=None):
        """Initialize the segmentation model.

        Inputs:
            weights_path: Path to model weights file (.onnx).
            config_path: Path to model config file (optional, for ONNX class names).
        """
        self.weights_path = weights_path
        print('[*] Loading segmentation model weights', weights_path)
        
        if not weights_path.endswith('.onnx'):
            raise ValueError("Only .onnx models are supported.")

        self.model = YOLOSegmentation(weights_path, config_path)

    def infer_masks(self, input_image):
        """Infer the segmentation masks in the input image.

        Inputs:
            input_image: Path to image or image array to detect food in.
        Returns:
            masks: [H,W,N] array containing each of the N masks detected.
            class_names: List of N class names.
        """
        # Load image if path given
        if isinstance(input_image, str):
            image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        else:
            image = input_image
            
        # Ensure RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Returns boxes, scores, class_ids, masks (H,W,N), class_names
        _, _, _, masks, class_names = self.model.predict(image_rgb)
        return masks, class_names

    def infer_and_plot(self, image_paths):
        """Infer the model output on a single image and plot the results.

        Inputs:
            image_paths: List of paths to images to detect food in.
        """
        import matplotlib.pyplot as plt

        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes, scores, class_ids, masks = self.model.predict(image_rgb)
            
            # Simple visualization
            plt.figure(figsize=(12, 10))
            plt.imshow(image_rgb)
            
            # Overlay masks
            if masks.shape[-1] > 0:
                mask_overlay = np.zeros_like(image_rgb)
                for i in range(masks.shape[-1]):
                    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                    mask = masks[:, :, i]
                    mask_overlay[mask > 0.5] = color
                
                plt.imshow(mask_overlay, alpha=0.5)
            
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Infer food segmentation masks using the YOLO ONNX model.')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.onnx',
                        help='Path to weights file.')
    parser.add_argument('--config', required=False,
                        metavar='/path/to/config.yaml',
                        help='Path to config file.')
    parser.add_argument('--images', required=False, nargs='+',
                        metavar='/path/1 /path/2 ...',
                        help='Path to one or more images to detect food in.')
    args = parser.parse_args()
    
    # Create segmentator object
    seg_model = FoodSegmentator(args.weights, args.config) 
    # Infer segmentation masks and plot results
    if args.images:
        seg_model.infer_and_plot(args.images)

