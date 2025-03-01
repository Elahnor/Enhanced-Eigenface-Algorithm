import cv2

def detect_movement(current_frame, previous_frame, min_area=2000):
    """Detect if the user's face is moving by comparing frames."""
    
    # Ensure that both frames are the same size
    if current_frame.shape != previous_frame.shape:
        previous_frame = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))

    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray_current, gray_previous)

    _, threshold_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            return True 

    return False
