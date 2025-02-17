import os
import cv2
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

def read_dataset(self):
    if self.recognize_face_btn.isChecked():
        try:                                       
            if self.enhanced_eigen_algo_radio.isChecked():
                # Load both recognizers for enhanced algorithm
                self.face_recognizer.read("training/enhanced_eigen_trained_dataset.yml")
                self.lbph_recognizer.read("training/lbph_trained_dataset.yml")
            else:
                self.face_recognizer.read("training/eigen_trained_dataset.yml")
        except Exception as e:
                pass

def recognize(self):       
    if self.recognize_face_btn.isChecked():
        trained_model_path = "training/eigen_trained_dataset.yml"
        enhanced_model_path = "training/enhanced_eigen_trained_dataset.yml"
        lbph_model_path = "training/lbph_trained_dataset.yml"
        
        original_dataset_path = "dataset/Original"
        enhanced_dataset_path = "dataset/Enhanced"
        
        if self.eigen_algo_radio.isChecked():
            if os.path.exists(original_dataset_path):
                if not os.path.exists(trained_model_path):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning) 
                    msg.setWindowTitle("Training Required")  
                    msg.setText("Please train the dataset before performing facial recognition.")  
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))  
                    msg.exec_()  
                    self.recognize_face_btn.setChecked(False)  
                    self.recognize_face_btn.setText("Recognize Face")  
                    return
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning) 
                msg.setWindowTitle("Dataset Not Found")  
                msg.setText("Please generate dataset and train dataset before performing facial recognition.")  
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setWindowIcon(QIcon("icon/AppIcon.png"))  
                msg.exec_()  
                self.recognize_face_btn.setChecked(False)  
                self.recognize_face_btn.setText("Recognize Face")  
                return
        
        elif self.enhanced_eigen_algo_radio.isChecked():
            if os.path.exists(enhanced_dataset_path):
                if not (os.path.exists(enhanced_model_path) and os.path.exists(lbph_model_path)):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning) 
                    msg.setWindowTitle("Training Required")  
                    msg.setText("Please train the dataset before performing facial recognition.")  
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))  
                    msg.exec_()  
                    self.recognize_face_btn.setChecked(False)  
                    self.recognize_face_btn.setText("Recognize Face")  
                    return
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning) 
                msg.setWindowTitle("Dataset Not Found")  
                msg.setText("Please generate dataset and train dataset before performing facial recognition.")  
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setWindowIcon(QIcon("icon/AppIcon.png"))  
                msg.exec_()  
                self.recognize_face_btn.setChecked(False)  
                self.recognize_face_btn.setText("Recognize Face")  
                return
        
        self.start_timer()
        self.recognize_face_btn.setText("Stop Recognition")
        self.read_dataset()
    else:
        self.recognize_face_btn.setText("Recognize Face")
        self.stop_timer()

        self.image = cv2.imread("icon/TitleScreen.png", 1)
        self.modified_image = self.image.copy()
        self.display()