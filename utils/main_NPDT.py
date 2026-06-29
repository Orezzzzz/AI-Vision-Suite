import cv2

from detector_NPDT import NumberPlateDetector
from ocr_NPDT import PlateOCR


class ANPRSystem:
    """
    This class combines number plate detection and OCR.
    It uses NumberPlateDetector to find plates and PlateOCR to read them.
    """

    def __init__(self, yolo_model_path, yaml_path):
        # Create detector object
        self.detector = NumberPlateDetector(yolo_model_path, yaml_path)

        # Create OCR object
        self.ocr = PlateOCR()

    def predict(self, image_path):
        """
        Detect number plates and read their text from an image.
        Returns the detection results and annotated image.
        """

        # Step 1: Read input image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found or could not be loaded: " + image_path)

        annotated_image = image.copy()

        # Step 2: Detect number plates
        boxes, confidences, class_ids, indices = self.detector.detect(image)

        results = []

        # Step 3: Crop every detected plate and apply OCR
        for i in indices:
            
            box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            x, y, w, h = box

            # Add a small padding around the plate crop
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)

            plate_crop = image[y1:y2, x1:x2]

            # Step 4: Read number plate text
            plate_text = self.ocr.read_plate(plate_crop)

            # Get class name from detector
            class_name = self.detector.labels[class_id]

            # Store result in a simple dictionary
            result = {
                "box": box,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "plate_text": plate_text
            }
            results.append(result)

            # Step 5: Draw bounding box
            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Step 6: Draw label with class name, confidence and OCR text
            label = class_name + ": " + str(round(confidence, 2))

            if plate_text != "":
                label = label + " | " + plate_text

            cv2.putText(
                annotated_image,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return results, annotated_image


if __name__ == "__main__":
    # Example usage
    anpr = ANPRSystem(
        yolo_model_path="models/best.onnx",
        yaml_path="data.yaml"
    )

    results, output_image = anpr.predict("car.jpg")

    # Print all results
    for result in results:
        print(result)

    # Save annotated output image
    cv2.imwrite("anpr_result.jpg", output_image)
