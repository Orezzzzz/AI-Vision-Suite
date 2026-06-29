import cv2
import numpy as np
import yaml


class PlantDiseaseDetector:
    """
    This class is for detecting plant diseases using a YOLO ONNX model.
    It performs image preprocessing, model inference, output decoding, and Non-Maximum-Supression.
    """

    def __init__(self, model_path, yaml_path, input_size=960):
        #Store basic settings
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.input_size = input_size

        #Thresholds can be changed depending on model performance
        self.confidence_threshold = 0.30
        self.nms_threshold = 0.40

        #Load class names from data.yaml
        with open(self.yaml_path, "r") as file:
            data = yaml.safe_load(file)
            self.labels = data["names"]
            self.nc = data['nc']

        #Load YOLO model using OpenCV DNN
        self.model = cv2.dnn.readNetFromONNX(self.model_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image):
        """
        Detect plant diseases in an input image.
        Returns boxes, confidence scores, and class ids, indices.
        """

        #Step 1: Get original image size
        original_height, original_width = image.shape[:2]

        #Step 2: Convert the image into a square image
        #YOLO expects a square input, so padding is added if needed
        max_side = max(original_height, original_width)
        square_image = np.zeros((max_side, max_side, 3), dtype=np.uint8)

        x_offset = (max_side - original_width) // 2
        y_offset = (max_side - original_height) // 2

        square_image[
            y_offset:y_offset + original_height,
            x_offset:x_offset + original_width
        ] = image

        #Step 3: Convert image to blob for YOLO
        blob = cv2.dnn.blobFromImage(
            square_image,
            1 / 255,
            (self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )

        #Step 4: Run inference
        self.model.setInput(blob)
        predictions = self.model.forward()

        #Some YOLO ONNX models return output as (1, channels, detections)
        #This converts it into (1, detections, channels)
        if predictions.shape[1] < predictions.shape[2]:
            predictions = predictions.transpose((0, 2, 1))

        detections = predictions[0]

        boxes = []
        confidences = []
        class_ids = []

        x_factor = max_side / self.input_size
        y_factor = max_side / self.input_size

        #Step 5: Decode all detections
        for detection in detections:
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > self.confidence_threshold:
                cx, cy, w, h = detection[0:4]

                #Convert YOLO center format to OpenCV rectangle format
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                #Remove the padding offset to get box position on original image
                left = left - x_offset
                top = top - y_offset

                #Keep the box inside the image
                left = max(0, left)
                top = max(0, top)
                width = max(0, min(width, original_width - left))
                height = max(0, min(height, original_height - top))

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

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
            
            x, y, w, h = boxes[i]
            
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
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        return output

