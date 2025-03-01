import os
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog
from utility.user_info import USER

dataset_per_subject = 20
current_path = None
previous_frame = None  

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
                ui.generate_dataset_btn.setText("Stop Generating")
                
                ui.current_path = current_path
                
            else:
                raise Exception("No algorithm selected")
        except:
            msg = QMessageBox(ui)
            msg.setWindowTitle("User Information")
            msg.setText("Provide the Proper Information. Please Try Again! \n\nName: (Any Combination of Letters and Numbers) \nKey: (Only numbers 0-9)")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            ui.generate_dataset_btn.setChecked(False)

