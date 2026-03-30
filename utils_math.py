import math

def calculate_angle(a, b, c):
    """
    Calculate the 2D angle between three joints using their (x, y) coordinates.
    a: First point (e.g. hip)
    b: Middle point (e.g. knee)
    c: Last point (e.g. ankle)
    Returns: angle in degrees mapped to 0-180.
    """
    # math.atan2(y, x) -> arc tangent in radians
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def calculate_percentage(angle, min_angle, max_angle):
    """
    Maps the current angle to a 0% - 100% score based on target range.
    For exercises where the target is a smaller angle (like a bicep curl where 
    160 is 0% and 30 is 100%):
        min_angle is the "100%" target (e.g., 30)
        max_angle is the "0%" starting point (e.g., 160)
        
    For exercises like reverse (e.g., standing up from squat):
        min_angle is 90 (100% squat depth)
        max_angle is 180 (0% standing)
    """
    if min_angle < max_angle:
        # e.g., Squat down to 90 degrees
        # smaller angle = higher percentage
        if angle <= min_angle:
            return 100.0
        if angle >= max_angle:
            return 0.0
        percentage = 100.0 * (1.0 - (angle - min_angle) / (max_angle - min_angle))
    else:
        # e.g. Raising arm (larger angle = higher percentage)
        if angle >= min_angle:
            return 100.0
        if angle <= max_angle:
            return 0.0
        percentage = 100.0 * (angle - max_angle) / (min_angle - max_angle)
        
    return max(0.0, min(100.0, percentage))

def get_landmark_array(landmark_obj):
    """ Helper to safely extract [x, y] from a MediaPipe landmark. """
    return [landmark_obj.x, landmark_obj.y]
