"""
Webcam detection utility.

This module contains a small class used by the Streamlit webcam pipeline.
It keeps the real-time frame processing separate from app.py, which makes the
project easier to understand and maintain.
"""

from __future__ import annotations

import threading

import numpy as np


class WebcamDetector:
    """
    Process webcam frames using an object detector.

    Parameters
    ----------
    detector:
        An ObjectDetector object. The object must provide:
        - detect(image)
        - draw_detections(image, boxes, confidences, class_ids, indices)
    """

    def __init__(self, detector):
        self.detector = detector

        # OpenCV DNN inference is protected by a lock because webcam frames may
        # be handled asynchronously by streamlit-webrtc.
        self.lock = threading.Lock()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on one webcam frame and return an annotated frame.

        The frame is expected to be a BGR image because OpenCV uses BGR format.
        """

        with self.lock:
            boxes, confidences, class_ids, indices = self.detector.detect(frame)

            annotated_frame = self.detector.draw_detections(
                frame,
                boxes,
                confidences,
                class_ids,
                indices,
            )

        return annotated_frame
