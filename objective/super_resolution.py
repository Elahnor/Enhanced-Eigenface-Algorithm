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

import cv2
import numpy as np

def is_real_face(face_image, min_face_size=(50, 50), max_face_size=(300, 300), min_brightness=20, max_brightness=500):
    if len(face_image.shape) == 3:
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_image

    sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    if sharpness < 70:
        return False  

    height, width = gray_face.shape
    if width < min_face_size[0] or height < min_face_size[1]:
        return False  
    if width > max_face_size[0] or height > max_face_size[1]:
        return False  
    
    avg_brightness = np.mean(gray_face)
    
    if avg_brightness < min_brightness or avg_brightness > max_brightness:
        return False 

    return True

def image_preprocess(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = denoise_image(image)
    image = adjust_contrast_clahe(image)
    image = sharpen_image(image)
    image = cubic_interpolation(image)
    return image
