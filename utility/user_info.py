import os
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi

class USER(QDialog):
    """USER Dialog"""
    def __init__(self):
        super(USER, self).__init__()
        loadUi("user_info.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = int(self.key_label.text())
        return name, key

    def validate_user_info(self, dataset_paths):
        """Validates if the user information already exists in the dataset paths."""
        name, key = self.get_name_key()

        if any(char.isdigit() for char in name):
            self.show_warning("Name should not contain any numbers.\nPlease enter a valid Name.")
            return False

        if not any(os.listdir(dataset_path) for dataset_path in dataset_paths):
            return True

        for dataset_path in dataset_paths:
            existing_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
            for folder in existing_folders:
                if folder.startswith(f"{key}-{name}"):
                    folder_path = os.path.join(dataset_path, folder)
                    existing_images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                    if len(existing_images) < 19:
                        return True
                    self.show_warning("Name and Key already exist. Please enter a different Name or Key.")
                    return False

                existing_name = folder.split("-")[1]
                existing_key = folder.split("-")[0]

                if existing_name.lower() == name.lower():
                    self.show_warning("Name already exists. Please enter a different Name.")
                    return False

                if existing_key == str(key):
                    self.show_warning("Key already exists. Please enter a different Key.")
                    return False

        return True

    def show_warning(self, message):
        """Displays a warning message box."""
        msg = QMessageBox(self)
        msg.setWindowTitle("User Information")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()