# image_processor.py

import cv2
import numpy as np
import easyocr  # Import EasyOCR

class ImageProcessor:
    """
    Handles all image processing tasks such as loading, filtering,
    and license plate detection using OCR.
    
    This version uses EasyOCR instead of Tesseract.
    """
    def __init__(self):
        # Initialize the EasyOCR reader. 
        # This will download the model the first time it's run.
        # We specify 'en' for English.
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, image: np.ndarray) -> tuple:
        """
        Applies a series of preprocessing steps to the image to prepare it for
        contour detection. (This function remains the same)
        """
        if image is None:
            return None, None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bilateral, 170, 200)
        return gray, bilateral, edged

    def find_license_plate_contour(self, edged_image: np.ndarray) -> np.ndarray:
        """
        Finds the contour that most likely represents the license plate.
        (This function remains the same)
        """
        if edged_image is None:
            return None
        contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        plate_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1.5 < aspect_ratio < 4.5:
                    plate_contour = approx
                    break
        return plate_contour

    def extract_plate(self, original_image: np.ndarray, plate_contour: np.ndarray) -> np.ndarray:
        """
        Extracts the license plate region from the original image using a mask.
        (This function remains the same)
        """
        if plate_contour is None or original_image is None:
            return None
        mask = np.zeros(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).shape, np.uint8)
        cv2.drawContours(mask, [plate_contour], 0, 255, -1)
        (x, y, w, h) = cv2.boundingRect(plate_contour)
        plate_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        cropped_plate = plate_image[y:y + h, x:x + w]
        return cropped_plate

    def recognize_text(self, plate_image: np.ndarray) -> str:
        """
        Uses EasyOCR to recognize text from the license plate image.
        
        Args:
            plate_image (np.ndarray): The cropped image of the license plate.

        Returns:
            str: The recognized text, cleaned and combined.
        """
        if plate_image is None:
            return "Detection Failed"

        try:
            # EasyOCR works directly on the BGR image.
            # The result is a list of tuples: (bounding_box, text, confidence_score)
            result = self.reader.readtext(plate_image)
            
            if not result:
                return "No Text Found"
            
            # Combine all detected text fragments into a single string
            # and filter out non-alphanumeric characters.
            full_text = " ".join([res[1] for res in result])
            return "".join(filter(str.isalnum, full_text)).upper()
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return "OCR Failed"