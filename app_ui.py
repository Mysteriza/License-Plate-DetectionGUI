# app_ui.py

import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.uic import loadUi
from image_processor import ImageProcessor

class DetectionThread(QThread):
    """
    A separate thread to run the license plate detection process,
    preventing the GUI from freezing.
    """
    finished = pyqtSignal(object, object, object, object, str)

    def __init__(self, image: np.ndarray, processor: ImageProcessor):
        super().__init__()
        self.image = image
        self.processor = processor

    def run(self):
        """The main work of the thread is done here."""
        original_image = self.image.copy()
        gray, bilateral, edged = self.processor.preprocess_image(original_image)
        
        plate_contour = self.processor.find_license_plate_contour(edged)
        
        if plate_contour is not None:
            cv2.drawContours(original_image, [plate_contour], -1, (0, 255, 0), 3)
            extracted_plate = self.processor.extract_plate(self.image.copy(), plate_contour)
            detected_text = self.processor.recognize_text(extracted_plate)
        else:
            extracted_plate = None
            detected_text = "Plate Not Found"
            
        self.finished.emit(gray, bilateral, edged, extracted_plate, detected_text)


class MainWindow(QMainWindow):
    """The main window of the application."""

    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('GUI.ui', self)
        self.setWindowTitle("License Plate Detector")

        self.image_processor = ImageProcessor()
        self.original_image = None
        self.detection_thread = None

        self.loadButton.clicked.connect(self.load_image)
        self.deteksiButton.clicked.connect(self.start_detection)

    def load_image(self):
        """Opens a file dialog to load an image."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.imgLabel)
            self.clear_result_labels()

    def start_detection(self):
        """Starts the license plate detection in a separate thread."""
        if self.original_image is None:
            self.OutputLabel.setText("Please load an image first.")
            return

        self.deteksiButton.setEnabled(False)
        self.OutputLabel.setText("Detecting...")

        self.detection_thread = DetectionThread(self.original_image.copy(), self.image_processor)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()

    def on_detection_finished(self, gray, bilateral, edged, extracted_plate, text):
        """
        This method is called when the detection thread finishes.
        It updates the GUI with the results.
        """
        self.display_image(gray, self.imgLabel_gray)
        self.display_image(bilateral, self.imgLabel_bilateral)
        self.display_image(edged, self.imgLabel_canny)
        
        if extracted_plate is not None:
            self.display_image(extracted_plate, self.hasilLabel)
            self.OutputLabel.setText(f"Detected Text: {text}")
        else:
            self.hasilLabel.setText("No Plate Found")
            self.OutputLabel.setText("Detection Failed")
        
        self.deteksiButton.setEnabled(True)

    def display_image(self, image: np.ndarray, label: QLabel):
        """
        Displays an OpenCV image (numpy array) on a PyQt QLabel.
        """
        if image is None:
            label.setText("N/A")
            return
            
        if len(image.shape) == 2:
            qformat = QImage.Format_Indexed8
        else:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        # FIX: Ensure the image data is in a contiguous memory block
        image = np.ascontiguousarray(image)
        
        img = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
        
        if len(image.shape) > 2 and image.shape[2] == 3:
            img = img.rgbSwapped()
        
        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)

    def clear_result_labels(self):
        """Clears all result labels."""
        self.imgLabel_gray.clear()
        self.imgLabel_bilateral.clear()
        self.imgLabel_canny.clear()
        self.hasilLabel.clear()
        self.OutputLabel.clear()