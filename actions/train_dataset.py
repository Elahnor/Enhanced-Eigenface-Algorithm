import os
import numpy as np
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer

from utility.get_info import get_labels_and_faces

def train_dataset(self):
    """Handles the training of the dataset."""
    if self.train_dataset_btn.isChecked():
        selected_algo_button = self.algo_radio_group.checkedButton()
        self.progress_bar_train.setValue(1)
        selected_algo_button.setEnabled(False)
        self.train_dataset_btn.setText("Stop Training")
        os.makedirs("training", exist_ok=True)

        original_dataset_path = "dataset/Original"
        enhanced_dataset_path = "dataset/Enhanced"

        if self.eigen_algo_radio.isChecked() and not os.path.exists(original_dataset_path):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Dataset Required")
            msg.setText("Please generate dataset before performing dataset training.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
            msg.exec_()
            self.train_dataset_btn.setChecked(False)
            self.train_dataset_btn.setText("Train Dataset")
            return

        if self.enhanced_eigen_algo_radio.isChecked() and not os.path.exists(enhanced_dataset_path):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Dataset Required")
            msg.setText("Please generate dataset before performing dataset training.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
            msg.exec_()
            self.train_dataset_btn.setChecked(False)
            self.train_dataset_btn.setText("Train Dataset")
            return

        dataset_folder = "Enhanced" if self.enhanced_eigen_algo_radio.isChecked() else "Original"
        labels, faces = get_labels_and_faces(dataset_folder)

        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Training Started")
            msg.setText("Dataset Training Started")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
            msg.exec_()

            self.progress_timer = QTimer()
            self.progress_timer.timeout.connect(lambda: update_progress(self))
            self.progress_timer.start(50)  # Update progress every 100ms

            if self.enhanced_eigen_algo_radio.isChecked():
                self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

                for i in range(1, 51):
                    self.face_recognizer.train(faces, np.array(labels))
                    self.progress_bar_train.setValue(i)

                for i in range(51, 101):
                    self.lbph_recognizer.train(faces, np.array(labels))
                    self.progress_bar_train.setValue(i)

            else:
                self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
                for i in range(1, 101):
                    self.face_recognizer.train(faces, np.array(labels))
                    self.progress_bar_train.setValue(i)

            self.progress_timer.stop()
            save_trained_dataset(self)

        except Exception as e:
            self.print_custom_error("Unable to Train the Dataset")
            print(f"Error during training: {e}")
            self.progress_timer.stop()
            self.progress_bar_train.setValue(0)
    else:
        self.eigen_algo_radio.setEnabled(True)
        self.enhanced_eigen_algo_radio.setEnabled(True)
        self.progress_bar_train.setValue(0)
        self.train_dataset_btn.setChecked(False)
        self.train_dataset_btn.setText("Train Dataset")

def update_progress(self):
    """Gradually updates progress bar to ensure a smooth transition."""
    current_value = self.progress_bar_train.value()
    if current_value < 100:
        self.progress_bar_train.setValue(current_value + 1)
    else:
        self.progress_timer.stop()

def save_trained_dataset(self):
    """Saves the trained dataset."""
    try:
        if self.enhanced_eigen_algo_radio.isChecked():
            self.face_recognizer.save("training/enhanced_eigen_trained_dataset.yml")
            self.lbph_recognizer.save("training/lbph_trained_dataset.yml")
        else:
            self.face_recognizer.save("training/eigen_trained_dataset.yml")
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Training Completed")
        msg.setText("Dataset Training Completed")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
        msg.exec_()
    except Exception as e:
        self.print_custom_error("Unable to Save Trained Dataset")
        print(f"Error during saving: {e}")
