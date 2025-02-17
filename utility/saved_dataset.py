import os
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from utility.calculation import calculate_distance
from utility.movement import detect_movement
from objective.super_resolution import image_preprocess, is_real_face  # Import the new validation function

previous_frame = None
dataset_per_subject = 20

def save_dataset(ui):
    """Saves dataset for each subject and handles movement detection."""
    global previous_frame, dataset_per_subject

    original_location = os.path.join(ui.current_path, str(dataset_per_subject) + ".png")
    enhanced_location = os.path.join(os.getcwd(), "dataset", "Enhanced", os.path.basename(ui.current_path), str(dataset_per_subject) + ".png")

    if ui.generate_dataset_btn.isChecked() == False and dataset_per_subject < 20:
        ui.generate_dataset_btn.setText("Generate Dataset")
        ui.generate_dataset_btn.setChecked(False)
        ui.stop_timer()
        dataset_per_subject = 20

        ui.image = cv2.imread("icon/TitleScreen.png", 1)
        ui.modified_image = ui.image.copy()
        ui.display()
        return  

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

        if ui.enhanced_eigen_algo_radio.isChecked() and previous_frame is not None and detect_movement(ui.image, previous_frame):
            ui.draw_text("Please Stop Moving!", 10, 30, color=(0, 0, 255))
            ui.draw_text("Generating Dataset in Progress.", 10, 60, color=(0, 0, 255))
            for (x, y, w, h) in faces:
                cv2.rectangle(ui.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            if len(faces) == 1:
                ui.draw_rectangle(faces)

            if len(faces) != 1:
                ui.draw_text("Please keep only one person visible.", 10, 30, color=(0, 0, 255))
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

                                # Face validation check
                                if not is_real_face(ui.image[y:y + h, x:x + w]):
                                    msg = QMessageBox()
                                    msg.setIcon(QMessageBox.Warning)
                                    msg.setText("Invalid Face Images. Please Try Again!")
                                    msg.setWindowTitle("Face Validation Failed")
                                    msg.setWindowIcon(QIcon("icon/AppIcon.png"))
                                    msg.setStandardButtons(QMessageBox.Ok)
                                    msg.exec_()
                                    # Stop the dataset generation process
                                    ui.generate_dataset_btn.setChecked(False)
                                    ui.generate_dataset_btn.setText("Generate Dataset")
                                    ui.stop_timer()
                                    ui.image = cv2.imread("icon/TitleScreen.png", 1)
                                    ui.modified_image = ui.image.copy()
                                    ui.display()
                                    return

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

    previous_frame = ui.image.copy()
    ui.display()