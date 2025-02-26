import cv2
import time
from utility.calculation import calculate_confidence_level, get_scaling_factor_for_distance
from utility.get_info import get_all_key_name_pairs

def distance_scaling(self, roi_gray_original, roi_color, distance, x, y, w, h, recognition_start_time, recognition_times):
    # Apply distance-based scaling for face image
    scaling_factor = get_scaling_factor_for_distance(distance)
    scaled_roi_gray = self.resize_image(roi_gray_original, int(600 * scaling_factor), int(600 * scaling_factor))

    expected_size = (600, 600) 
    
    if self.eigen_algo_radio.isChecked():
        expected_size = (300, 300) 
    scaled_roi_gray = cv2.resize(scaled_roi_gray, expected_size)

    if self.recognize_face_btn.isChecked() and (self.eigen_algo_radio.isChecked() or self.enhanced_eigen_algo_radio.isChecked()):
        try:
            prediction_start_time = time.time()
            predicted, _ = self.face_recognizer.predict(scaled_roi_gray)
            name = get_all_key_name_pairs().get(str(predicted))

            self.draw_text(name, x - 5, y - 5)

            prediction_end_time = time.time()
            recognition_time = round(prediction_end_time - prediction_start_time, 4)

            # Recognition Time
            if self.recog_time_checkbox.isChecked() and (30 <= distance <= 60): 
                total_recognition_time = round(time.time() - recognition_start_time, 4)
                cv2.putText(self.image, f"Recognition Time: {total_recognition_time}s", (10, 30), \
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)

            # Confidence Level
            if self.predict_confidence_checkbox.isChecked():
                if self.enhanced_eigen_algo_radio.isChecked():
                    confidence_level = calculate_confidence_level(distance, enhanced=True)
                else:
                    confidence_level = calculate_confidence_level(distance, enhanced=False)
                confidence_text = f"Confidence Level: {confidence_level:.2f}%"
                cv2.putText(self.image, confidence_text, (10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, \
                            color=(0, 255, 0), thickness=2)
                print(f"Recognition Time: {recognition_time}s, Distance: {distance} cm, Confidence Level: {confidence_level:.2f}")

            recognition_times.append(recognition_time)

        except Exception as e:
            self.print_custom_error("Unable to Predict due to")
            print(e)

    if self.eye_rect_radio.isChecked():
        eyes = self.get_eyes(roi_gray_original)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    elif self.smile_rect_radio.isChecked():
        smiles = self.get_smiles(roi_gray_original)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    else:
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if not self.enhanced_eigen_algo_radio.isChecked() or (30 <= distance <= 60):
        distance_text = f"{distance} cm"
        self.draw_text(distance_text, x + 50, y + h + 25, color=(0, 255, 0))

    return recognition_times