import cv2
import numpy as np
import onnxruntime as ort
import yaml
import os

class YOLOSegmentation:
    def __init__(self, model_path, config_path=None, conf_thres=0.25, iou_thres=0.45, mask_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.mask_thres = mask_thres
        
        # Load Model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape # [1, 3, 640, 640] usually
        self.img_size = self.input_shape[2]
        
        # Load Classes
        self.classes = []
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                self.classes = data.get('names', [])
        
    def preprocess(self, img):
        self.original_shape = img.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize
        img_data = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_data = img_data.transpose(2, 0, 1)
        
        # Add batch dim
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def predict(self, img):
        """
        Runs inference on the image.
        Returns:
            boxes: [N, 4] (x1, y1, x2, y2)
            scores: [N]
            class_ids: [N]
            masks: [H, W, N] binary masks (0 or 1)
            class_names: List of N class names
        """
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        return self.postprocess(outputs)

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape
        ih, iw = shape
        
        # Matmul: [n, 32] @ [32, 160*160] -> [n, 160*160]
        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = 1 / (1 + np.exp(-masks)) 
        
        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        
        crop_masks = []
        for i, box in enumerate(downsampled_bboxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mw, x2), min(mh, y2)
            
            mask = masks[i]
            full_mask = np.zeros((mh, mw), dtype=np.float32)
            if x2 > x1 and y2 > y1:
                full_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
            if upsample:
                full_mask = cv2.resize(full_mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
                
            crop_masks.append(full_mask)
            
        if not crop_masks:
             return np.zeros((0, ih, iw), dtype=np.float32)

        return np.array(crop_masks) # [N, H, W]

    def postprocess(self, outputs):
        preds = outputs[0][0].transpose() # [8400, 139]
        protos = outputs[1][0] # [32, 160, 160]
                           
        box_preds = preds[:, :4] # [8400, 4]
        
        # Assuming typical YOLOv8 Seg output: Box(4) + Class(NC) + Mask(32)
        # Verify indices if classes differ.
        # Check num classes
        nc = len(self.classes) if self.classes else 103 # Default to 103 if not loaded or fallback
        # Or deduce from shape: 139 - 4 - 32 = 103. Correct.
        
        class_scores = preds[:, 4:4+nc]
        mask_coeffs = preds[:, 4+nc:]
        
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        mask = confidences > self.conf_thres
        box_preds = box_preds[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        mask_coeffs = mask_coeffs[mask]
        
        if len(confidences) == 0:
            return [], [], [], np.zeros((self.original_shape[0], self.original_shape[1], 0)), []

        boxes_xyxy = self.xywh2xyxy(box_preds)
        
        boxes_tlwh = box_preds.copy()
        boxes_tlwh[:, 0] = box_preds[:, 0] - box_preds[:, 2] / 2
        boxes_tlwh[:, 1] = box_preds[:, 1] - box_preds[:, 3] / 2
        
        indices = cv2.dnn.NMSBoxes(boxes_tlwh.tolist(), confidences.tolist(), self.conf_thres, self.iou_thres)
        
        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = boxes_xyxy[indices]
            final_scores = confidences[indices]
            final_cls_ids = class_ids[indices]
            final_masks_coeffs = mask_coeffs[indices]
            
            # Generate masks
            # Note: process_mask returns [N, H, W]
            final_masks = self.process_mask(protos, final_masks_coeffs, final_boxes, (self.img_size, self.img_size), upsample=True)
            
            # Resize masks to original image shape and Threshold
            resized_masks = []
            for m in final_masks:
                # Resize to (W_orig, H_orig)
                m_res = cv2.resize(m, (self.original_shape[1], self.original_shape[0]), interpolation=cv2.INTER_LINEAR)
                m_bin = (m_res > self.mask_thres).astype(np.float32)
                resized_masks.append(m_bin)
            
            if not resized_masks:
                 return [], [], [], np.zeros((self.original_shape[0], self.original_shape[1], 0)), []

            # Stack to [H, W, N]
            masks_stack = np.stack(resized_masks, axis=-1)
            
            # Scale boxes to original size
            scale_x = self.original_shape[1] / self.img_size
            scale_y = self.original_shape[0] / self.img_size
            
            final_boxes[:, 0] *= scale_x
            final_boxes[:, 2] *= scale_x
            final_boxes[:, 1] *= scale_y
            final_boxes[:, 3] *= scale_y
            
            # Get class names
            final_class_names = []
            if self.classes:
                for cls_id in final_cls_ids:
                    if cls_id < len(self.classes):
                        final_class_names.append(self.classes[cls_id])
                    else:
                        final_class_names.append(str(cls_id))
            else:
                 final_class_names = [str(c) for c in final_cls_ids]

            return final_boxes, final_scores, final_cls_ids, masks_stack, final_class_names
        else:
            return [], [], [], np.zeros((self.original_shape[0], self.original_shape[1], 0)), []
