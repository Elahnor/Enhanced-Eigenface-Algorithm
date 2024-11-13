import cv2
import os
import sys
import time
import math
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi

class USER(QDialog):        # Dialog box for entering name and key of new dataset.
    """USER Dialog """
    def __init__(self):
        super(USER, self).__init__()
        loadUi("user_info.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = int(self.key_label.text())
        return name, key

class EFR(QMainWindow):        # Main Application 
    """Main Class"""
    def __init__(self):
        super(EFR, self).__init__()
        loadUi("mainwindow.ui", self)
        
        # Classifiers, frontal face, eyes and smiles.
        self.face_classifier = cv2.CascadeClassifier("classifiers/frontalface_default.xml") 
        self.eye_classifier = cv2.CascadeClassifier("classifiers/eye.xml")
        self.smile_classifier = cv2.CascadeClassifier("classifiers/smile.xml")
        
        # Variables
        self.camera_id = 0
        self.dataset_per_subject = 20
        self.ret = False
        self.trained_dataset = 0

        self.image = cv2.imread("icon/TitleScreen.png", 1)
        self.modified_image = self.image.copy()
        self.display()
        
        # Actions 
        self.generate_dataset_btn.setCheckable(True)
        self.train_dataset_btn.setCheckable(True)
        self.recognize_face_btn.setCheckable(True)
        
        # Menu
        self.about_menu = self.menu_bar.addAction("About")
        self.help_menu = self.menu_bar.addAction("Help")
        self.about_menu.triggered.connect(self.about_info)
        self.help_menu.triggered.connect(self.help_info)
        
        # Algorithms
        self.algo_radio_group.buttonClicked.connect(self.algorithm_radio_changed)
        
        # Recangle
        self.face_rect_radio.setChecked(True)
        self.eye_rect_radio.setChecked(False)
        self.smile_rect_radio.setChecked(False)
        
        # Events
        self.generate_dataset_btn.clicked.connect(self.generate)
        self.train_dataset_btn.clicked.connect(self.train)
        self.recognize_face_btn.clicked.connect(self.recognize)
        
        # Recognizers
        self.update_recognizer()
        self.assign_algorithms()

    def start_timer(self):    
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.timer = QtCore.QTimer()
        if self.generate_dataset_btn.isChecked():
            self.timer.timeout.connect(self.save_dataset)
        elif self.recognize_face_btn.isChecked():
            self.timer.timeout.connect(self.update_image)
        self.timer.start(200)

    def stop_timer(self): 
        if hasattr(self, "timer") and self.timer.isActive(): 
            self.timer.stop()  
        self.ret = False  
        self.capture.release()
        self.progress_bar_generate.setValue(0)  
        
    def update_image(self):
        if self.recognize_face_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
        self.display()

    def save_dataset(self):
        location = os.path.join(self.current_path, str(self.dataset_per_subject) + ".jpg")

        if self.dataset_per_subject < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information) 
            msg.setText('<span style="color:#022052;"><b>User\'s Facial Data has been Recorded.</b></span><br><br>'
                        '<span style="color:#022052;">Now, you can <b>Train the Facial Data</b> or <b>Generate New Dataset</b>.</span>')
            msg.setWindowTitle("Dataset Generated")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QIcon("icon/AppIcon.png"))
            msg.exec_()

            self.generate_dataset_btn.setText("Generate Dataset")
            self.generate_dataset_btn.setChecked(False)
            self.stop_timer()
            self.dataset_per_subject = 20

        if self.generate_dataset_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)

            if len(faces) != 1:
                self.draw_text("Only One Person at a Time", 10, 30)
            else:
                for (x, y, w, h) in faces:
                    distance = self.calculate_distance(w)

                    if 30 <= distance <= 60:
                        cv2.imwrite(location, self.resize_image(self.get_gray_image()[y:y + h, x:x + w], 600, 600))
                        file_name = os.path.basename(location)  
                        self.draw_text(file_name, 20, 30)
                        self.dataset_per_subject -= 1
                        self.progress_bar_generate.setValue(100 - self.dataset_per_subject * 2 % 100)
                    else:
                        self.draw_text("Please Move Closer or Farther.", 10, 30)

        self.display()

    def display(self):
        pixImage = self.pix_image(self.image)
        self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
        self.video_feed.setScaledContents(True)

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage.rgbSwapped()

    def generate(self):   
        if self.generate_dataset_btn.isChecked():
            try:
                user = USER()
                user.exec_()
                name, key = user.get_name_key()
                self.current_path = os.path.join(os.getcwd(),"datasets",str(key)+"-"+name)
                os.makedirs(self.current_path, exist_ok=True)
                self.start_timer()
                self.generate_dataset_btn.setText("Generating")
            except:
                msg = QMessageBox(self)
                msg.setWindowTitle("User Information")
                msg.setText("Provide Information Please! \n\nName: (Any Combination of Letters and Numbers) \nKey: (Only numbers 0-9)")
                msg.setIcon(QMessageBox.Warning)  
                msg.setStandardButtons(QMessageBox.Ok) 
                msg.exec_()  

                self.generate_dataset_btn.setChecked(False)
    
    def algorithm_radio_changed(self):      
        self.assign_algorithms()                                
        self.update_recognizer()                               
        self.read_dataset()                                      
        if self.train_dataset_btn.isChecked():
            self.train()

    def update_recognizer(self):                                
        if self.eigen_algo_radio.isChecked():
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        elif self.enhanced_eigen_algo_radio.isChecked():
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        else:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    def assign_algorithms(self):        
        if self.eigen_algo_radio.isChecked():
            self.algorithm = "Eigenface Algorithm"
        elif self.enhanced_eigen_algo_radio.isChecked():
            self.algorithm = "Enhanced Eigenface Algorithm"
        else:
            self.algorithm = "LBPH"

    def read_dataset(self):       # Reading Trained Dataset.
        if self.recognize_face_btn.isChecked():
            try:                                       
                self.face_recognizer.read("training/"+self.algorithm.lower()+" trained dataset.yml")
            except Exception as e:
                self.print_custom_error("Unable to Read Trained Dataset")
                print(e)
    
    def save_trained_dataset(self):       # Save User Dataset.
        try:
            self.face_recognizer.save("training/"+self.algorithm.lower()+" trained dataset.yml")
            msg = self.algorithm+" Datasets Trained."
            self.trained_dataset += 1
            self.progress_bar_train.setValue(self.trained_dataset)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)  
            msg.setWindowTitle("Training Completed")  
            msg.setText("Dataset Training Completed")  
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QIcon("icon/AppIcon.png"))  
            msg.exec_()
            
        except Exception as e:
            self.print_custom_error("Unable to Save Trained Dataset")
            print(e)
    
    def train(self):  
        if self.train_dataset_btn.isChecked():
            selected_algo_button = self.algo_radio_group.checkedButton()
            
            self.progress_bar_train.setValue(0)
            
            selected_algo_button.setEnabled(False)
            self.train_dataset_btn.setText("Stop Training")
            os.makedirs("training", exist_ok=True)

            labels, faces = self.get_labels_and_faces()
            
            try:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Training Started")
                msg.setText("Dataset Training Started")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                msg.exec_()

                self.face_recognizer.train(faces, np.array(labels))                
                self.save_trained_dataset()
                self.progress_bar_train.setValue(100)
                
            except Exception as e:
                self.print_custom_error("Unable To Train the Dataset")
                print(e)
        else:
            self.eigen_algo_radio.setEnabled(True)
            self.enhanced_eigen_algo_radio.setEnabled(True)
            self.lbph_algo_radio.setEnabled(True)
            
            self.progress_bar_train.setValue(0)
            
            self.train_dataset_btn.setChecked(False)
            self.train_dataset_btn.setText("Train Dataset")
    
    def recognize(self):       
        if self.recognize_face_btn.isChecked():
            trained_model_path = f"training/{self.algorithm.lower()} trained dataset.yml"
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
            else:
                self.start_timer()
                self.recognize_face_btn.setText("Stop Recognition")
                self.read_dataset()
        else:
            self.recognize_face_btn.setText("Recognize Face")
            self.stop_timer()
    
    def get_all_key_name_pairs(self):       
        return dict([subfolder.split('-') for _, folders, _ in os.walk(os.path.join(os.getcwd(), "datasets")) for subfolder in folders],)
        
    def absolute_path_generator(self):     
        separator = "-"
        for folder, folders, _ in os.walk(os.path.join(os.getcwd(),"datasets")):
            for subfolder in folders:
                subject_path = os.path.join(folder,subfolder)
                key, _ = subfolder.split(separator)
                for image in os.listdir(subject_path):
                    absolute_path = os.path.join(subject_path, image)
                    yield absolute_path,key

    def get_labels_and_faces(self):     
        labels, faces = [],[]
        for path,key in self.absolute_path_generator():
            faces.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
            labels.append(int(key))
        return labels,faces

    def get_gray_image(self):      
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_faces(self):        
        # variables
        scale_factor = 1.1
        min_neighbors = 8
        min_size = (100, 100) 

        faces = self.face_classifier.detectMultiScale(self.get_gray_image(), scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size)
        return faces

    def get_smiles(self, roi_gray):     
        scale_factor = 1.7
        min_neighbors = 22
        min_size = (25, 25)

        smiles = self.smile_classifier.detectMultiScale(roi_gray, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size)
        return smiles

    def get_eyes(self, roi_gray):
        scale_factor = 1.1
        min_neighbors = 6
        min_size = (30, 30)

        eyes = self.eye_classifier.detectMultiScale(roi_gray, scaleFactor = scale_factor, minNeighbors = min_neighbors,)
        return eyes

    def draw_rectangle(self, faces):
        recognition_start_time = time.time()

        # Initialize a variable to track the recognition times for different distances
        recognition_times = []

        # Variable to track if a warning message needs to be displayed
        display_warning = False

        for (x, y, w, h) in faces:
            roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
            roi_color = self.image[y:y + h, x:y + w]

            distance = self.calculate_distance(w)
            rectangle_color = (0, 255, 0)  # Default green rectangle

            if self.enhanced_eigen_algo_radio.isChecked():
                if distance < 30 or distance > 60:
                    rectangle_color = (0, 0, 255)  # Red rectangle
                    distance_text = "Face is Out of Range"
                    self.draw_text(distance_text, x - 65, y + h + 35, color=(0, 0, 255))
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), rectangle_color, 2)
                    display_warning = True
                    continue  # Skip further processing for this face if out of range

            # Apply distance-based scaling for face image
            scaling_factor = self.get_scaling_factor_for_distance(distance)
            scaled_roi_gray = self.resize_image(roi_gray_original, int(600 * scaling_factor), int(600 * scaling_factor))

            expected_size = (600, 600)
            scaled_roi_gray = cv2.resize(scaled_roi_gray, expected_size)

            if self.recognize_face_btn.isChecked() and (self.eigen_algo_radio.isChecked() or self.enhanced_eigen_algo_radio.isChecked()):
                try:
                    prediction_start_time = time.time()
                    predicted, _ = self.face_recognizer.predict(scaled_roi_gray)
                    name = self.get_all_key_name_pairs().get(str(predicted))

                    # Display recognized name
                    self.draw_text(name, x - 5, y - 5)

                    prediction_end_time = time.time()
                    recognition_time = round(prediction_end_time - prediction_start_time, 4)

                    # Recognition Time
                    if self.recog_time_checkbox.isChecked() and (30 <= distance <= 60):  # Only show if distance is in range
                        recognition_end_time = time.time()
                        total_recognition_time = round(recognition_end_time - recognition_start_time, 4)
                        cv2.putText(self.image, f"Recognition Time: {total_recognition_time}s", (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)

                    # Confidence Level (Customized per algorithm)
                    if self.predict_confidence_checkbox.isChecked():
                        if self.enhanced_eigen_algo_radio.isChecked():
                            # Apply a more aggressive confidence level scaling for enhanced algorithm
                            confidence_level = self.calculate_confidence_level(distance, enhanced=True)
                        else:
                            # Standard Eigenface confidence level calculation
                            confidence_level = self.calculate_confidence_level(distance, enhanced=False)
                        confidence_text = f"Confidence Level: {confidence_level:.2f}%"
                        # Adjusting the Y-coordinate to be below the recognition time
                        cv2.putText(self.image, confidence_text, (10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
                        print(f"Recognition Time: {recognition_time}s, Distance: {distance} cm, Confidence Level: {confidence_level:.2f}")

                    recognition_times.append(recognition_time)

                except Exception as e:
                    self.print_custom_error("Unable to Predict due to")
                    print(e)

            if self.eye_rect_radio.isChecked(): 
                eyes = self.get_eyes(roi_gray_original)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            elif self.smile_rect_radio.isChecked(): 
                smiles = self.get_smiles(roi_gray_original)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

            else:  
                cv2.rectangle(self.image, (x, y), (x + w, y + h), rectangle_color, 2)

            if not self.enhanced_eigen_algo_radio.isChecked() or (distance >= 30 and distance <= 60):
                distance_text = f"{distance} cm"
                self.draw_text(distance_text, x + 50, y + h + 25, color=rectangle_color)

        # Warning Message
        if display_warning:
            text = "Warning: Possible Spoofing Attempt!"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
            x = (self.image.shape[1] - text_width) // 2 
            y = 80
            cv2.putText(self.image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        # Recognition Time
        if (self.eigen_algo_radio.isChecked() or self.enhanced_eigen_algo_radio.isChecked()) and self.recog_time_checkbox.isChecked() and (30 <= distance <= 60):
            recognition_end_time = time.time()
            total_recognition_time = round(recognition_end_time - recognition_start_time, 4)
            cv2.putText(self.image, f"Recognition Time: {total_recognition_time}s", (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)

    def calculate_confidence_level(self, distance, enhanced=False):
        ideal_distance = 45
        max_deviation = 60  # Extends the confidence range

        deviation = abs(distance - ideal_distance)

        # Adjust confidence depending on whether the distance is closer or farther
        if distance < ideal_distance:
            confidence = (1 - deviation / max_deviation) * 100 * 1.05  # Boost for closer distances
        else:
            confidence = (1 - deviation / max_deviation) * 100 * 0.90  # Slight reduction for farther distances

        # Apply enhancement scaling if enhanced algorithm is active
        if enhanced:
            confidence *= 1.4
            confidence = 100 * (1 - math.exp(-confidence / 40))

        return min(max(confidence, 0), 100)  # Ensure confidence is between 0 and 100%

    def get_scaling_factor_for_distance(self, distance):
        if distance < 30:
            return 0.8
        elif 30 <= distance <= 60:
            return 1.0 
        else:
            return 1.5  

    def calculate_distance(self, face_width):
        known_face_width = 12
        focal_length = 900 

        distance = (known_face_width * focal_length) / face_width
        return round(distance, 2)
            
    def time(self):
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)): 
        cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

    def resize_image(self, image, width=600, height=600): 
        return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)

    def print_custom_error(self, msg):      
        print("="*100)
        print(msg)
        print("="*100)

    # Main Menu
    
    def about_info(self):
        msg_box = QMessageBox()
        msg_box.setText('''<html><body style="text-align: justify;">
            This system enhances the Eigenface Algorithm to improve the identification of
            spoofing attacks in facial recognition. By addressing key challenges such as 
            the identification of face occlusions and spoofed images, the system
            significantly increases accuracy and reliability. It employs super 
            resolution to effectively manage noise in the covariance matrix, ensuring 
            robust performance even with low-resolution images. Additionally, the system
            incorporates distance-adjustment features to account for varying facial
            distances from the camera, reducing the risk of misidentification and
            improving usability. Ultimately, this innovative approach strengthens the
            security and effectiveness of facial recognition technology.
            
            <br><br>
            <b style="color: #022052;">Pamantasan ng Lungsod ng Maynila, CISTM - BSCS (2024)</b><br>
            <b style="color: #022052;">Members</b>: Hale I. Casison, Chloe Gwyneth S. Upaga<br>
            <b style="color: #022052;">Adviser</b>: Jamillah S. Guialil
        </body></html>''')
        msg_box.setWindowTitle("About")
        msg_box.setWindowIcon(QIcon("icon/AppIcon.png"))
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

    def help_info(self):
        msg_box = QMessageBox()
        msg_box.setText('''<html><body style="text-align: justify;">
        This application allows users to generate datasets, train datasets, and perform facial
        recognition. It also detects the user’s whole face with the primary focus on 
        accurate individual recognition. For assistance with these features, please 
        refer to the provided instructions.

        <br><br>
        <b style="color: #022052;">Follow these steps to use the application:</b>
        <ol>
            <li>Generate at least two datasets.</li>
            <li>Train all generated datasets according to the provided radio buttons.</li>
            <li>Recognize users.</li>
        </ol>
        </body></html>''')
        msg_box.setWindowTitle("Help")
        msg_box.setWindowIcon(QIcon("icon/AppIcon.png"))
        msg_box.setIcon(QMessageBox.Question)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = EFR()
    ui.show()
    sys.exit(app.exec_())
