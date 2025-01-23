import math

def calculate_confidence_level(distance, enhanced=False):
    ideal_distance = 45
    max_deviation = 60 

    deviation = abs(distance - ideal_distance)

    if distance < ideal_distance:
        confidence = (1 - deviation / max_deviation) * 100 * 1.05
    else:
        confidence = (1 - deviation / max_deviation) * 100 * 0.90 

    if enhanced:
        confidence *= 1.4
        confidence = 100 * (1 - math.exp(-confidence / 40))

    return min(max(confidence, 0), 100)

def get_scaling_factor_for_distance(distance):
    if distance < 30:
        return 0.8
    elif 30 <= distance <= 60:
        return 1.0
    else:
        return 1.5

def calculate_distance(face_width):
    known_face_width = 12
    focal_length = 900 

    distance = (known_face_width * focal_length) / face_width
    return round(distance, 2)