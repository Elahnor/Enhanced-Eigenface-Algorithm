import os
import cv2
import time
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox, QLabel
from objective.super_resolution import is_real_face
from utility.calculation import calculate_confidence_level, get_scaling_factor_for_distance
from utility.get_info import get_all_key_name_pairs

def resize_image_for_display(image, width=200):
    aspect_ratio = image.shape[1] / image.shape[0]
    height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def distance_scaling(self, roi_gray_original, roi_color, distance, x, y, w, h, recognition_start_time, recognition_times):
    scaling_factor = get_scaling_factor_for_distance(distance)
    scaled_roi_gray = self.resize_image(roi_gray_original, int(600 * scaling_factor), int(600 * scaling_factor))

    expected_size = (600, 600) 
    
    if self.eigen_algo_radio.isChecked():
        expected_size = (300, 300) 
    scaled_roi_gray = cv2.resize(scaled_roi_gray, expected_size)

    if self.enhanced_eigen_algo_radio.isChecked():
        faces = self.get_faces()

        if len(faces) != 1:
            self.draw_text("Please ensure only one person is visible.", 10, 30, color=(0, 0, 255))
            return recognition_times

    if self.recognize_face_btn.isChecked() and (self.eigen_algo_radio.isChecked() or self.enhanced_eigen_algo_radio.isChecked()):
        try:
            if self.enhanced_eigen_algo_radio.isChecked():
                if not is_real_face(roi_color):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning) 
                    msg.setText("<b><font color='red'>Invalid Facial Recognition. Potential Spoofing Attack is IDENTIFIED.</font></b>")
                    msg.setWindowTitle("Facial Recognition Failed")
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                    msg.setStandardButtons(QMessageBox.Ok)  

                    full_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) 
                    resized_image = resize_image_for_display(full_image, width=200)

                    height, width, channel = resized_image.shape
                    bytes_per_line = 3 * width
                    qimage = QPixmap.fromImage(QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888))
                    
                    face_label = QLabel()
                    face_label.setPixmap(qimage)
                    msg.setInformativeText("<font color='red'>Note: This image illustrates the cause of the facial recognition failure which is due to a potential spoofing attack using a fake or altered face.</font>")
                    msg.setIconPixmap(qimage)
                    
                    msg.exec_()

                    self.recognize_face_btn.setChecked(False)
                    self.recognize_face_btn.setText("Recognize Face")

                    self.stop_timer()
                    self.image = cv2.imread("icon/TitleScreen.png", 1)
                    self.modified_image = self.image.copy()
                    self.display()
                    return recognition_times

            prediction_start_time = time.time()
            predicted, _ = self.face_recognizer.predict(scaled_roi_gray)
            name = get_all_key_name_pairs().get(str(predicted))

            self.draw_text(name, x - 5, y - 5)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not self.enhanced_eigen_algo_radio.isChecked() or (30 <= distance <= 60):
                distance_text = f"{distance} cm"
                self.draw_text(distance_text, x + 50, y + h + 25, color=(0, 255, 0))

            self.display()
            time.sleep(0.5)

            if name:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Successful Facial Recognition")
                msg.setText(f"\nHello {name}!\n\n\nNote: If you wish to recognize another face, click 'Recognize Face' again to start a new session")
                msg.setStyleSheet("color: #022052;")

                face_label = QLabel()
                recognized_face = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                resized_face = resize_image_for_display(recognized_face, width=200)

                height, width, channel = resized_face.shape
                bytes_per_line = 3 * width
                qimage = QPixmap.fromImage(QImage(resized_face.data, width, height, bytes_per_line, QImage.Format_RGB888))
                face_label.setPixmap(qimage)
                msg.setDetailedText(name)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                msg.setIconPixmap(qimage)  
                
                response = msg.exec_()
                
                if response == QMessageBox.Ok:
                    self.stop_timer()
                    self.image = cv2.imread("icon/TitleScreen.png", 1)
                    self.modified_image = self.image.copy()
                    self.display()
                    self.recognize_face_btn.setChecked(False)
                    self.recognize_face_btn.setText("Recognize Face")
                    return recognition_times

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

    return recognition_times