import cv2
import sys
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

from utility.menu_info import MenuInfo
from actions.generate_dataset import generate
from actions.train_dataset import train_dataset
from actions.recognize_face import read_dataset, recognize
from utility.saved_dataset import save_dataset
from utility.get_info import get_labels_and_faces, get_gray_image, get_faces
from utility.select_algo import algorithm_radio_changed, update_recognizer, assign_algorithms
from utility.about_image import display, update_image, pix_image, resize_image
from utility.timer import Timer
from utility.display_info import draw_text, draw_rectangle

class EFR(QMainWindow):  
    """Main Class"""
    def __init__(self):
        super(EFR, self).__init__()
        loadUi("mainwindow.ui", self)
        
        # Classifier for frontal face
        self.face_classifier = cv2.CascadeClassifier("classifiers/frontalface_default.xml") 
        
        # Variables
        self.camera_id = 0
        self.dataset_per_subject = 20
        self.ret = False
        self.trained_dataset = 0
        self.current_path = None

        self.image = cv2.imread("icon/TitleScreen.png", 1)
        self.modified_image = self.image.copy()
        self.display()
        
        # Main Menu
        self.about_menu = self.menu_bar.addAction("About")
        self.help_menu = self.menu_bar.addAction("Help")
        self.about_menu.triggered.connect(self.about_info)
        self.help_menu.triggered.connect(self.help_info)
        
        # Actions 
        self.generate_dataset_btn.setCheckable(True)
        self.train_dataset_btn.setCheckable(True)
        self.recognize_face_btn.setCheckable(True)
        
        # Algorithms
        self.algo_radio_group.buttonClicked.connect(self.algorithm_radio_changed)
        self.assign_algorithms()
        
        # Timer instance
        self.timer_manager = Timer(self)
       
        # Events
        self.generate_dataset_btn.clicked.connect(self.generate)
        self.train_dataset_btn.clicked.connect(self.train)
        self.recognize_face_btn.clicked.connect(self.recognize)
        
        # Recognizers
        self.update_recognizer()
        self.assign_algorithms()
        
    #MAIN MENU
    def about_info(self):
        MenuInfo.about_info()

    def help_info(self):
        MenuInfo.help_info()

    def print_custom_error(self, msg):      
        print("="*100)
        print(msg)
        print("="*100)
            
    #ACTIONS
    def generate(self):
        generate(self)
    
    def save_dataset(self):
        save_dataset(self)  # Call save_dataset from saved_dataset.py
    
    def train(self):
        train_dataset(self)

    def recognize(self):
        recognize(self)

    def read_dataset(self):
        read_dataset(self)
    
    #GET INFORMATION
    def get_labels_and_faces(self):  
        labels, faces = get_labels_and_faces()
        return labels, faces

    def get_gray_image(self):  
        return get_gray_image(self.image)
    
    def get_faces(self):  
        faces = get_faces(self.image, self.face_classifier)
        return faces
        
    #ALGORITHMS SECTION
    def algorithm_radio_changed(self):      
        algorithm_radio_changed(self)

    def update_recognizer(self):                                 
        update_recognizer(self)
        
    def assign_algorithms(self):        
        assign_algorithms(self)
            
    #TIMER
    def start_timer(self):
        self.timer_manager.start_timer(self.camera_id)

    def stop_timer(self):
        self.timer_manager.stop_timer()

    def time(self):
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")
            
    #ABOUT IMAGE
    def display(self):
        display(self)

    def update_image(self):
        update_image(self)

    def pix_image(self, image):
        return pix_image(self, image)

    def resize_image(self, image, width=300, height=300):
        return resize_image(self, image, width, height)

    #DISPLAY INFORMATION
    def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)): 
        draw_text(self.image, text, x, y, font_size, color)
        
    def draw_rectangle(self, faces):
        draw_rectangle(self, self.image, faces, self.enhanced_eigen_algo_radio, self.eigen_algo_radio)
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = EFR()
    ui.show()
    sys.exit(app.exec_())
