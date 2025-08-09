# License Plate Detection with Python and OpenCV

This project is a graphical user interface (GUI) application for detecting and recognizing license plates from images. It is built using Python, OpenCV, and PyQt5.

## Features

- Load images in various formats (JPG, PNG, etc.).
- Pre-processes the image using Grayscale, Bilateral Filtering, and Canny Edge Detection.
- Automatically finds the license plate in the image.
- Uses Tesseract OCR to read the characters on the plate.
- Non-blocking GUI that remains responsive during detection.

## Prerequisites

- Python 3.6+
- Tesseract OCR Engine

### Tesseract Installation

You must install Google's Tesseract OCR engine on your system.

- **Windows**: Download the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. Make sure to note the installation path.
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update
  sudo apt install tesseract-ocr
  ```
