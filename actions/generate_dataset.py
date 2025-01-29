import os
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from utility.calculation import calculate_distance
from utility.user_info import USER
from objective.super_resolution import image_preprocess

dataset_per_subject = 20
current_path = None

def generate(ui):
    """Handles dataset generation"""
    global current_path, dataset_per_subject

    if ui.generate_dataset_btn.isChecked():
        try:
            user = USER()
            user.exec_()
            name, key = user.get_name_key()
            current_path = os.path.join(os.getcwd(), "dataset", "Original", str(key) + "-" + name)
            enhanced_path = os.path.join(os.getcwd(), "dataset", "Enhanced", str(key) + "-" + name)
            os.makedirs(current_path, exist_ok=True)
            os.makedirs(enhanced_path, exist_ok=True)
            ui.start_timer()
            ui.generate_dataset_btn.setText("Generating")
        except:
            msg = QMessageBox(ui)
            msg.setWindowTitle("User Information")
            msg.setText("Provide Information Please! \n\nName: (Any Combination of Letters and Numbers) \nKey: (Only numbers 0-9)")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            ui.generate_dataset_btn.setChecked(False)

def save_dataset(ui):
    """Saves dataset for each subject"""
    global current_path, dataset_per_subject

    original_location = os.path.join(current_path, str(dataset_per_subject) + ".png")
    enhanced_location = os.path.join(os.getcwd(), "dataset", "Enhanced", os.path.basename(current_path), str(dataset_per_subject) + ".png")

    if dataset_per_subject < 1:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText('<span style="color:#022052;"><b>User\'s Facial Data has been Recorded.</b></span><br><br>'
                    '<span style="color:#022052;">Now, you can <b>Train the Facial Data</b> or <b>Generate New Dataset</b>.</span>')
        msg.setWindowTitle("Dataset Generated")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowIcon(QIcon("icon/AppIcon.png"))
        msg.exec_()

        ui.generate_dataset_btn.setText("Generate Dataset")
        ui.generate_dataset_btn.setChecked(False)
        ui.stop_timer()
        dataset_per_subject = 20

    if ui.generate_dataset_btn.isChecked():
        ui.ret, ui.image = ui.capture.read()
        ui.image = cv2.flip(ui.image, 1)
        faces = ui.get_faces()
        ui.draw_rectangle(faces)

        if len(faces) != 1:
            ui.draw_text("Only One Person at a Time", 10, 30)
        else:
            for (x, y, w, h) in faces:
                distance = calculate_distance(w)

                if 30 <= distance <= 60:
                    gray_image = ui.get_gray_image()[y:y + h, x:x + w]
                    resized_image = ui.resize_image(gray_image, 300, 300)
                    enhanced_image = image_preprocess(resized_image)

                    cv2.imwrite(original_location, resized_image)
                    cv2.imwrite(enhanced_location, enhanced_image)

                    file_name = os.path.basename(original_location)
                    ui.draw_text(file_name, 20, 30)
                    dataset_per_subject -= 1
                    ui.progress_bar_generate.setValue(100 - dataset_per_subject * 2 % 100)
                else:
                    ui.draw_text("Please Move Closer or Farther.", 10, 30)

    ui.display()