import os
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog
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
            if ui.eigen_algo_radio.isChecked() or ui.enhanced_eigen_algo_radio.isChecked():
                user = USER()
                result = user.exec_()

                if result == QDialog.Rejected:
                    ui.generate_dataset_btn.setChecked(False)
                    return

                if ui.eigen_algo_radio.isChecked():
                    dataset_paths = [os.path.join(os.getcwd(), "dataset", "Sample")]
                else:
                    dataset_paths = [
                        os.path.join(os.getcwd(), "dataset", "Original"),
                        os.path.join(os.getcwd(), "dataset", "Enhanced")
                    ]
                    
                for path in dataset_paths:
                    os.makedirs(path, exist_ok=True)

                if not user.validate_user_info(dataset_paths):
                    ui.generate_dataset_btn.setChecked(False)
                    return
                
                name, key = user.get_name_key()

                if ui.eigen_algo_radio.isChecked():
                    current_path = os.path.join(os.getcwd(), "dataset", "Sample", str(key) + "-" + name)
                else:
                    current_path = os.path.join(os.getcwd(), "dataset", "Original", str(key) + "-" + name)
                    enhanced_path = os.path.join(os.getcwd(), "dataset", "Enhanced", str(key) + "-" + name)
                    os.makedirs(enhanced_path, exist_ok=True)

                os.makedirs(current_path, exist_ok=True)

                ui.start_timer()
                ui.generate_dataset_btn.setText("Generating")
            else:
                raise Exception("No algorithm selected")
        except:
            msg = QMessageBox(ui)
            msg.setWindowTitle("User Information")
            msg.setText("Select an Algorithm and Provide Information! \n\nName: (Any Combination of Letters and Numbers) \nKey: (Only numbers 0-9)")
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

        ui.image = cv2.imread("icon/TitleScreen.png", 1)
        ui.modified_image = ui.image.copy()
        ui.display()

    if ui.generate_dataset_btn.isChecked():
        ui.ret, ui.image = ui.capture.read()
        ui.image = cv2.flip(ui.image, 1)
        faces = ui.get_faces()
        ui.draw_rectangle(faces)

        if len(faces) != 1:
            ui.draw_text("No face found. Keep one person visible.", 10, 30, color=(0, 0, 255))
        else:
            distance = None
            if ui.enhanced_eigen_algo_radio.isChecked():
                for (x, y, w, h) in faces:
                    distance = calculate_distance(w)
                    if not (30 <= distance <= 60):
                        cv2.rectangle(ui.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        ui.draw_text("Please Move Closer or Farther.", 10, 30, color=(0, 0, 255))

            if len(faces) == 1 and (distance is None or (30 <= distance <= 60)):
                ui.draw_text("Please Stay Still!", 10, 30, color=(0, 255, 0)) 

                if ui.enhanced_eigen_algo_radio.isChecked():
                    for (x, y, w, h) in faces:
                        if 30 <= distance <= 60:
                            gray_image = ui.get_gray_image()[y:y + h, x:x + w]
                            resized_image = ui.resize_image(gray_image, 300, 300)
                            enhanced_image = image_preprocess(resized_image)

                            cv2.imwrite(original_location, resized_image)
                            cv2.imwrite(enhanced_location, enhanced_image)

                            file_name = f"Saving {os.path.basename(original_location)}"
                            ui.draw_text(file_name, 10, 60)
                            dataset_per_subject -= 1

                            total_images = 20
                            progress_value = int(((total_images - dataset_per_subject) / total_images) * 100)
                            ui.progress_bar_generate.setValue(progress_value)
                else:
                    for (x, y, w, h) in faces:
                        gray_image = ui.get_gray_image()[y:y + h, x:x + w]
                        resized_image = ui.resize_image(gray_image, 300, 300)
                        cv2.imwrite(original_location, resized_image)
                        file_name = f"Saving {os.path.basename(original_location)}"
                        ui.draw_text(file_name, 10, 60)
                        dataset_per_subject -= 1

                        total_images = 20
                        progress_value = int(((total_images - dataset_per_subject) / total_images) * 100)
                        ui.progress_bar_generate.setValue(progress_value)
    
    ui.display()