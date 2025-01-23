from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

class USER(QDialog):      
    """USER Dialog """
    def __init__(self):
        super(USER, self).__init__()
        loadUi("user_info.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = int(self.key_label.text())
        return name, key
