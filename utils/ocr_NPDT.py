import cv2
import os
import re
import tempfile
from paddleocr import PaddleOCR


class PlateOCR:
    """
    This class performs OCR on cropped number plate images.
    PaddleOCR is used to read the characters from the plate.
    """

    def __init__(self):
        # Load PaddleOCR model
        # The try-except is used because PaddleOCR versions may have different arguments
        try:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False
            )
        except TypeError:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)

    def read_plate(self, plate_image):
        """
        Read text from one cropped number plate image.
        Returns cleaned text.
        """

        if plate_image is None or plate_image.size == 0:
            return ""

        # Step 1: Improve the plate crop before OCR
        processed_image = self.improve_image(plate_image)

        # Step 2: Save crop temporarily because PaddleOCR works well with image paths
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        cv2.imwrite(temp_path, processed_image)

        # Step 3: Run PaddleOCR
        try:
            if hasattr(self.ocr, "predict"):
                ocr_result = self.ocr.ocr(temp_path)
                text = self.extract_text_from_new_result(ocr_result)
            else:
                ocr_result = self.ocr.ocr(temp_path, cls=True)
                text = self.extract_text_from_old_result(ocr_result)
        finally:
            os.remove(temp_path)

        # Step 4: Clean OCR text
        text = self.clean_text(text)

        return text

    def improve_image(self, image):
        """
        Upscale and sharpen the cropped plate image.
        This helps OCR detect small plate characters better.
        """

        height, width = image.shape[:2]

        # If the plate is very small, increase its size more
        scale = 3
        if height < 80:
            scale = 4

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        # Reduce noise but keep edges
        gray = cv2.bilateralFilter(gray, 7, 45, 45)

        # Sharpen the image
        blur = cv2.GaussianBlur(gray, (0, 0), 1)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        return sharp

    def extract_text_from_new_result(self, result):
        """
        Extract text from newer PaddleOCR predict() output.
        """

        final_text = ""

        for item in result:
            data = getattr(item, "json", item)

            if callable(data):
                data = data()

            if isinstance(data, dict):
                data = data.get("res", data)
                texts = data.get("rec_texts", [])
                final_text += "".join(texts)

        return final_text

    def extract_text_from_old_result(self, result):
        """
        Extract text from older PaddleOCR ocr() output.
        """

        final_text = ""

        if result is None:
            return final_text

        lines = result[0] if len(result) == 1 else result

        for line in lines:
            if len(line) >= 2:
                text_and_score = line[1]
                if len(text_and_score) >= 1:
                    final_text += text_and_score[0]

        return final_text

    def clean_text(self, text):
        """
        Keep only uppercase letters and numbers.
        This is useful for number plate text.
        """

        text = text.upper()
        text = re.sub("[^A-Z0-9]", "", text)
        return text
