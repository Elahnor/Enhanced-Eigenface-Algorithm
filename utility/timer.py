import cv2
from PyQt5 import QtCore

class Timer:
    def __init__(self, parent):
        self.parent = parent
        self.timer = QtCore.QTimer()

    def start_timer(self, camera_id):
        self.parent.capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.parent.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.parent.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

        if self.parent.generate_dataset_btn.isChecked():
            self.timer.timeout.connect(self.parent.save_dataset)
        elif self.parent.recognize_face_btn.isChecked():
            self.timer.timeout.connect(self.parent.update_image)

        self.timer.start(200)

    def stop_timer(self):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        self.parent.ret = False
        self.parent.capture.release()
        self.parent.progress_bar_generate.setValue(0)
