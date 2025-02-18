from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon

class MenuInfo:
    @staticmethod
    def about_info():
        msg_box = QMessageBox()
        msg_box.setText('''<html><body style="text-align: justify;">
            This system enhances the Eigenface Algorithm to improve the identification of
            spoofing attacks in facial recognition. By addressing key challenges such as 
            the identification of face occlusions and spoofed images, the system
            significantly increases accuracy and reliability. 
            
            <br><br>
            It employs super resolution to effectively manage noise in the covariance matrix, ensuring 
            robust performance even with low-resolution images. Additionally, the system
            incorporates distance-adjustment features to account for varying facial
            distances from the camera, reducing the risk of misidentification and
            improving usability. Ultimately, this innovative approach strengthens the
            security and effectiveness of facial recognition technology.
        </body></html>''')
        msg_box.setWindowTitle("About")
        msg_box.setWindowIcon(QIcon("icon/AppIcon.png"))
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

    @staticmethod
    def help_info():
        msg_box = QMessageBox()
        msg_box.setText('''<html><body style="text-align: justify;">
        This application allows users to generate datasets, train datasets, and perform facial
        recognition. It also detects the user’s whole face with the primary focus on 
        accurate individual recognition. For assistance with these features, please 
        refer to the provided instructions.

        <br><br>
        <b style="color: #022052;">Follow these steps to use the application:</b>
        <ol>
            <li>Generate at least two datasets.</li>
            <li>Train all generated datasets according to the provided radio buttons.</li>
            <li>Recognize users.</li>
        </ol>
        </body></html>''')
        msg_box.setWindowTitle("Help")
        msg_box.setWindowIcon(QIcon("icon/AppIcon.png"))
        msg_box.setIcon(QMessageBox.Question)
        msg_box.exec_()
