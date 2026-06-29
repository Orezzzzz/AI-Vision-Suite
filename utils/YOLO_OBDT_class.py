
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
import os

class ObjectDetector:
    """
    This class is for detecting objects using a YOLO ONNX model.
    It performs image preprocessing, model inference, output decoding, and Non-Maximum-Supression.
    """
    
    def __init__(self,model_path,yaml_path,input_size=640):
        #storing settings
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.input_size = input_size
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.45
        
        # Load classes from data.yaml
        with open(self.yaml_path,mode='r') as f:
            data_yaml = yaml.load(f,Loader = SafeLoader)
            self.labels = data_yaml['names']
            self.nc = data_yaml['nc']

        #Load YOLO model using OpenCV DNN
        self.yolo = cv2.dnn.readNetFromONNX(self.model_path)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self,image):
        """
        Detect objects in an input image.
        Returns boxes, confidence scores, and class ids.
        """
        #Step 1: Get original image size
        original_height,original_width, d = image.shape
        
        #Get YOLO predictions from the image
        #Step 1: convert the image into a square array
        max_side = max(original_height,original_width)
        input_image = np.zeros((max_side,max_side,3),dtype = np.uint8)
        input_image[0:original_height,0:original_width] = image
        
        
        #Step 3: Convert image to blob for YOLO
        blob = cv2.dnn.blobFromImage(
            input_image,
            1/255,
            (self.input_size,self.input_size),
            swapRB = True,
            crop = False
        )

        #Step 4: Run inference
        self.yolo.setInput(blob)
        predictions = self.yolo.forward()
        
        #Some YOLO ONNX models return output as (1, channels, detections)
        #This converts it into (1, detections, channels)
        if predictions.shape[1] < predictions.shape[2]:
            predictions = predictions.transpose((0, 2, 1))    
        
        detections = predictions[0]
        
        boxes = []
        confidences = []
        class_ids = []
        
        image_h, image_w = input_image.shape[:2]
        
        x_factor = max_side / self.input_size
        y_factor = max_side / self.input_size
        
        #Step 5: Decode all detections
        for det in detections:
        
            classes_scores = det[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]
        
            if confidence > self.confidence_threshold:
                cx, cy, w, h = det[0:4]
                
                #Convert YOLO center format to OpenCV rectangle format
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
        
                boxes.append([left, top, width, height])
        
                confidences.append(float(confidence))
        
                class_ids.append(class_id)
        
        #Step 6: Apply NMS
        if len(boxes) == 0:
            return [], [], [], []
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []
            
        return boxes, confidences, class_ids, indices
        

    #Draw bounding boxes and labels on image.
    def draw_detections(
            self,
            image,
            boxes,
            confidences,
            class_ids,
            indices
            ):
   
        output = image.copy()
            
        for i in indices:

            i = int(i)
        
            try:
                x, y, w, h = boxes[i]

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
            
            except Exception as e:
                raise ValueError(
                    f"boxes[{i}] = {boxes[i]} | "
                    f"type={type(boxes[i])} | "
                    f"error={e}"
                )
        
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
        
            label = self.labels[class_ids[i]]
        
            conf = confidences[i]
        
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"{label}: {conf*100:.2f}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        return output
