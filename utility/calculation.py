import math

def calculate_snr(signal_power, noise_power):
    if noise_power == 0:
        return float('inf') 
    
    snr = 10 * math.log10(signal_power / noise_power)
    return round(snr, 2)

def calculate_confidence_level(distance, face_image=None, enhanced=False):

    if face_image is None:
        face_image = [[100]]
    signal_power = sum([sum(row) for row in face_image]) / (len(face_image) * len(face_image[0])) 
    noise_power = 10

    snr = calculate_snr(signal_power, noise_power)
    ideal_distance = 45 
    max_deviation = 60 
    deviation = abs(distance - ideal_distance)

    if distance < ideal_distance:
        confidence = (1 - deviation / max_deviation) * 100 * 1.05 
    else:
        confidence = (1 - deviation / max_deviation) * 100 * 0.90  

    SNR_THRESHOLD = 20 
    if snr >= SNR_THRESHOLD:
        confidence *= 1.2  
    else:
        confidence *= 0.8 

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
