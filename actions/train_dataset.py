import os
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox

from utility.get_info import get_labels_and_faces

def train_dataset(self):
    """Handles the training of the dataset."""
    if self.train_dataset_btn.isChecked():
        selected_algo_button = self.algo_radio_group.checkedButton()
        self.progress_bar_train.setValue(0)
        selected_algo_button.setEnabled(False)
        self.train_dataset_btn.setText("Training Dataset")
        os.makedirs("training", exist_ok=True)

        labels, faces = get_labels_and_faces()

        try:
            # Notify user that training has started
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Training Started")
            msg.setText("Dataset Training Started")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
            msg.exec_()

            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            
            # Save trained dataset
            save_trained_dataset(self)
            self.progress_bar_train.setValue(100)
        
        except Exception as e:
            self.print_custom_error("Unable to Train the Dataset")
            print(e)
    else:
        # Reset buttons and progress bar
        self.eigen_algo_radio.setEnabled(True)
        self.enhanced_eigen_algo_radio.setEnabled(True)
        self.progress_bar_train.setValue(0)
        self.train_dataset_btn.setChecked(False)
        self.train_dataset_btn.setText("Train Dataset")

def save_trained_dataset(self):
    """Saves the trained dataset."""
    try:
        self.face_recognizer.save("training/" + self.algorithm.lower() + " trained dataset.yml")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Training Completed")
        msg.setText("Dataset Training Completed")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowIcon(QtGui.QIcon("icon/AppIcon.png"))
        msg.exec_()
    except Exception as e:
        self.print_custom_error("Unable to Save Trained Dataset")
        print(e)