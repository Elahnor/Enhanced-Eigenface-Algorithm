import os
import cv2

def get_all_key_name_pairs(dataset_folder="Original"):
    #Returns a dictionary of key-name pairs from the dataset folder structure
    return dict([subfolder.split('-') for _, folders, _ in os.walk(os.path.join(os.getcwd(), "Dataset", dataset_folder)) for subfolder in folders])

def absolute_path_generator(dataset_folder="Original"):
    #Generates the absolute paths for images and their corresponding keys
    separator = "-"
    for folder, folders, _ in os.walk(os.path.join(os.getcwd(), "Dataset", dataset_folder)):
        for subfolder in folders:
            subject_path = os.path.join(folder, subfolder)
            key, _ = subfolder.split(separator)
            for image in os.listdir(subject_path):
                absolute_path = os.path.join(subject_path, image)
                yield absolute_path, key

def get_labels_and_faces(dataset_folder="Original"):
    #Retrieves the labels and corresponding grayscale face images
    labels, faces = [], []
    for path, key in absolute_path_generator(dataset_folder):
        faces.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
        labels.append(int(key))
    return labels, faces

def get_gray_image(image):
    #Converts the image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_faces(image, face_classifier):
    #Detects faces in the given image
    scale_factor = 1.1
    min_neighbors = 8
    min_size = (100, 100)

    return face_classifier.detectMultiScale(get_gray_image(image), scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

def get_smiles(roi_gray, smile_classifier):
    #Detects smiles in the region of interest (ROI)
    scale_factor = 1.7
    min_neighbors = 22
    min_size = (25, 25)

    return smile_classifier.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

def get_eyes(roi_gray, eye_classifier):
    #Detects eyes in the region of interest (ROI)
    scale_factor = 1.1
    min_neighbors = 6
    min_size = (30, 30)

    return eye_classifier.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)