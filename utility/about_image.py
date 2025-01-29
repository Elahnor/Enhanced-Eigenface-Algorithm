import cv2
from PyQt5 import QtGui

def display(self):
    pixImage = self.pix_image(self.image)
    self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
    self.video_feed.setScaledContents(True)

def update_image(self):
    if self.recognize_face_btn.isChecked():
        self.ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        faces = self.get_faces()
        self.draw_rectangle(faces)
    self.display()

def pix_image(self, image):
    qformat = QtGui.QImage.Format_RGB888  # only RGB Image
    if len(image.shape) >= 3:
        r, c, ch = image.shape
    else:
        r, c = image.shape
        qformat = QtGui.QImage.Format_Indexed8
    pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
    return pixImage.rgbSwapped()

def resize_image(self, image, width=300, height=300): 
    return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)
