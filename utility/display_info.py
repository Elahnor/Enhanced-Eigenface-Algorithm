import cv2
import time
from objective.distance_scaling import distance_scaling
from utility.calculation import calculate_distance

def draw_text(image, text, x=20, y=20, font_size=2, color=(0, 255, 0)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

def draw_rectangle(self, image, faces, enhanced_eigen_algo_radio, eigen_algo_radio, recog_time_checkbox):
    recognition_start_time = time.time()

    recognition_times = []
    display_warning = False

    for (x, y, w, h) in faces:
        roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Ensure that distance is calculated before use
        distance = calculate_distance(w)

        if distance is None:
            continue
        
        #Green Rectangle
        rectangle_color = (0, 255, 0)

        if self.recognize_face_btn.isChecked() and enhanced_eigen_algo_radio.isChecked():
            if distance < 30 or distance > 60:
                rectangle_color = (0, 0, 255)  #Red Rectangle
                distance_text = "Face is Out of Range"
                draw_text(image, distance_text, x - 65, y + h + 35, color=(0, 0, 255))
                cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)
                display_warning = True
                continue  

        # Apply distance-based scaling and recognition
        recognition_times = distance_scaling(
            self, roi_gray_original, roi_color, distance, x, y, w, h, recognition_start_time, recognition_times
        )

    # Warning Message
    if display_warning:
        text = "Warning: Possible Spoofing Attempt!"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
        x = (image.shape[1] - text_width) // 2
        y = 80
        cv2.putText(image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    # Recognition Time
    if (eigen_algo_radio.isChecked() or enhanced_eigen_algo_radio.isChecked()) and recog_time_checkbox.isChecked():
        if eigen_algo_radio.isChecked():
            total_recognition_time = round(time.time() - recognition_start_time, 4)
            cv2.putText(image, f"Recognition Time: {total_recognition_time}s", (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
        elif (30 <= distance <= 60):
            total_recognition_time = round(time.time() - recognition_start_time, 4)
            cv2.putText(image, f"Recognition Time: {total_recognition_time}s", (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
