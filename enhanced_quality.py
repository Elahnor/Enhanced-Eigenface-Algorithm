import cv2
import numpy as np

def denoise_image(image):
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

def adjust_contrast_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1.2, gaussian, -0.2, 0)

def enhance_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = denoise_image(image)
    image = adjust_contrast_clahe(image)
    image = sharpen_image(image)

    return image
