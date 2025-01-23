import os
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

def read_dataset(self):
    if self.recognize_face_btn.isChecked():
        try:                                       
            self.face_recognizer.read("training/"+self.algorithm.lower()+" trained dataset.yml")
        except Exception as e:
            self.print_custom_error("Unable to Read Trained Dataset")
            print(e)

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
