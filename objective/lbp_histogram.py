import cv2
import numpy as np

def train_lbph(faces, labels):
    """Train the LBPH face recognizer."""
    lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
    lbph_recognizer.train(faces, np.array(labels))
    return lbph_recognizer

def save_lbph_model(lbph_recognizer, model_path):
    """Save the trained LBPH model."""
    lbph_recognizer.save(model_path)

def read_lbph_model(model_path):
    """Read the saved LBPH model."""
    lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
    lbph_recognizer.read(model_path)
    return lbph_recognizer

def is_real_face(face_image, min_face_size=(50, 50), max_face_size=(300, 300), min_brightness=20, max_brightness=150):
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

def detect_occlusion(face_image, eye_cascade, nose_cascade, mouth_cascade):
    """Detect occlusion by checking if both eyes, mouth, and nose are sufficiently visible."""
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray_face, 1.3, 5)
    nose = nose_cascade.detectMultiScale(gray_face, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray_face, 1.3, 5)

    min_eyes = 1 
    min_mouth = 1  
    min_nose = 1   

    if len(eyes) < min_eyes or len(mouth) < min_mouth or len(nose) < min_nose:
        return True  
    
    return False  
