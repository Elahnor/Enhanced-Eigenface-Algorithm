import os
import cv2
import time
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from objective.super_resolution import is_real_face
from utility.calculation import calculate_confidence_level, get_scaling_factor_for_distance
from utility.get_info import get_all_key_name_pairs

def distance_scaling(self, roi_gray_original, roi_color, distance, x, y, w, h, recognition_start_time, recognition_times):
    # Apply distance-based scaling for face image
    scaling_factor = get_scaling_factor_for_distance(distance)
    scaled_roi_gray = self.resize_image(roi_gray_original, int(600 * scaling_factor), int(600 * scaling_factor))

    expected_size = (600, 600) 
    
    if self.eigen_algo_radio.isChecked():
        expected_size = (300, 300) 
    scaled_roi_gray = cv2.resize(scaled_roi_gray, expected_size)

    # Check if the Enhanced Eigenface Algorithm is selected
    if self.enhanced_eigen_algo_radio.isChecked():
        # Get the number of faces detected
        faces = self.get_faces()

        # If more than one face is detected, display a message and return
        if len(faces) != 1:
            self.draw_text("Please ensure only one person is visible.", 10, 30, color=(0, 0, 255))
            return recognition_times

    if self.recognize_face_btn.isChecked() and (self.eigen_algo_radio.isChecked() or self.enhanced_eigen_algo_radio.isChecked()):
        try:
            # Check if the dataset is trained
            trained_model_path = "training/eigen_trained_dataset.yml"
            enhanced_model_path = "training/enhanced_eigen_trained_dataset.yml"
            lbph_model_path = "training/lbph_trained_dataset.yml"

            if self.enhanced_eigen_algo_radio.isChecked():
                if not (os.path.exists(enhanced_model_path) and os.path.exists(lbph_model_path)):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Training Required")
                    msg.setText("Please train the dataset first before recognizing faces.")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                    msg.exec_()
                    self.recognize_face_btn.setChecked(False)
                    self.recognize_face_btn.setText("Recognize Face")
                    # Stop the camera and display the TitleScreen image
                    self.stop_timer()
                    self.image = cv2.imread("icon/TitleScreen.png", 1)
                    self.modified_image = self.image.copy()
                    self.display()
                    return recognition_times
            else:
                if not os.path.exists(trained_model_path):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Training Required")
                    msg.setText("Please train the dataset first before recognizing faces.")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                    msg.exec_()
                    self.recognize_face_btn.setChecked(False)
                    self.recognize_face_btn.setText("Recognize Face")
                    # Stop the camera and display the TitleScreen image
                    self.stop_timer()
                    self.image = cv2.imread("icon/TitleScreen.png", 1)
                    self.modified_image = self.image.copy()
                    self.display()
                    return recognition_times

            # Check if the face is real using the enhanced Eigenface algorithm
            if self.enhanced_eigen_algo_radio.isChecked():
                if not is_real_face(roi_color):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Invalid Face Images. Please Try Again!")
                    msg.setWindowTitle("Face Validation Failed")
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    # Stop the recognition process entirely
                    self.recognize_face_btn.setChecked(False)
                    self.recognize_face_btn.setText("Recognize Face")
                    # Reset camera and image
                    self.stop_timer()
                    self.image = cv2.imread("icon/TitleScreen.png", 1)
                    self.modified_image = self.image.copy()
                    self.display()
                    return recognition_times

            prediction_start_time = time.time()
            predicted, _ = self.face_recognizer.predict(scaled_roi_gray)
            name = get_all_key_name_pairs().get(str(predicted))

            self.draw_text(name, x - 5, y - 5)

            prediction_end_time = time.time()
            recognition_time = round(prediction_end_time - prediction_start_time, 4)

            # Recognition Time
            if self.recog_time_checkbox.isChecked() and (30 <= distance <= 60): 
                total_recognition_time = round(time.time() - recognition_start_time, 4)
                cv2.putText(self.image, f"Recognition Time: {total_recognition_time}s", (10, 30), \
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)

            # Confidence Level
            if self.predict_confidence_checkbox.isChecked():
                if self.enhanced_eigen_algo_radio.isChecked():
                    confidence_level = calculate_confidence_level(distance, enhanced=True)
                else:
                    confidence_level = calculate_confidence_level(distance, enhanced=False)
                confidence_text = f"Confidence Level: {confidence_level:.2f}%"
                cv2.putText(self.image, confidence_text, (10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, \
                            color=(0, 255, 0), thickness=2)
                print(f"Recognition Time: {recognition_time}s, Distance: {distance} cm, Confidence Level: {confidence_level:.2f}")

            recognition_times.append(recognition_time)

        except Exception as e:
            self.print_custom_error("Facial Recognition Failed: Dataset Not Trained")
            print(e)

    # Draw rectangle around the face
    cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display distance text
    if not self.enhanced_eigen_algo_radio.isChecked() or (30 <= distance <= 60):
        distance_text = f"{distance} cm"
        self.draw_text(distance_text, x + 50, y + h + 25, color=(0, 255, 0))

    return recognition_times