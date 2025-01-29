import cv2

def algorithm_radio_changed(self):      
    self.assign_algorithms()                                
    self.update_recognizer()                               
    self.read_dataset()                                      
    if self.train_dataset_btn.isChecked():
        self.train()

def update_recognizer(self):                                
    if self.eigen_algo_radio.isChecked():
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
    elif self.enhanced_eigen_algo_radio.isChecked():
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()

def assign_algorithms(self):        
    if self.eigen_algo_radio.isChecked():
        self.algorithm = "Eigenface Algorithm"
    elif self.enhanced_eigen_algo_radio.isChecked():
        self.algorithm = "Enhanced Eigenface Algorithm"
