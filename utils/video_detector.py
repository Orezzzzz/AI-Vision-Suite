import cv2
import numpy as np

class VideoDetector:

    def __init__(self, detector):
        self.detector = detector

    def process_video(
        self,
        input_video_path,
        output_video_path="output.mp4"):
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {input_video_path}"
            )

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30
            
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        print("FPS:", fps)
        print("Width:", width)
        print("Height:", height)
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break
            if not writer.isOpened():
                raise ValueError("VideoWriter could not be opened")

            # Run detection
            boxes, confidences, class_ids, indices = (
                self.detector.detect(frame)
            )

            # Draw detections
            output_frame = (
                self.detector.draw_detections(
                    frame,
                    boxes,
                    confidences,
                    class_ids,
                    indices
                )
            )

            # Save frame
            writer.write(output_frame)

        cap.release()
        writer.release()

        return output_video_path