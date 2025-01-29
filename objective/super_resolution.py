import cv2

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

def image_preprocess(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = denoise_image(image)
    image = adjust_contrast_clahe(image)
    image = sharpen_image(image)
    image = cubic_interpolation(image)

    return image