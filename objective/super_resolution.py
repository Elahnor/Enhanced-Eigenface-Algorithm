# super_resolution.py

import cv2
import numpy as np

def denoise_image(image):
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

def adjust_contrast_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 1)
    return cv2.addWeighted(image, 1.2, gaussian, -0.2, 0)

def cubic_interpolation(image, scale_factor=2):
    height, width = image.shape[:2]
    new_dim = (width * scale_factor, height * scale_factor)
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)

def is_real_face(face_image):
    """
    A basic check if the detected face is a real face or not.
    In this example, we'll use simple criteria based on image sharpness.
    """
    # Check if the image is already in grayscale
    if len(face_image.shape) == 3:
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_image  # It's already grayscale

    # Calculate the sharpness of the image (using Laplacian variance)
    sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()  # A measure of sharpness

    # A sharpness threshold that can help detect if the face is printed/phone image
    if sharpness < 100:  # Arbitrary threshold, can be adjusted
        return False  # It is not a real face (likely a printed image or screen)
    return True  # It's a real face

def image_preprocess(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = denoise_image(image)
    image = adjust_contrast_clahe(image)
    image = sharpen_image(image)
    image = cubic_interpolation(image)
    return image
