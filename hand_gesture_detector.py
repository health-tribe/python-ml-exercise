import cv2
import time
import math

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe Tasks API not found. Please ensure mediapipe is installed correctly.")
    exit()

# Global variables for async callback results from MediaPipe
latest_result = None

def get_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def detect_gesture(landmarks):
    """
    Super basic heuristic-based gesture detection based on finger states.
    Uses the new MediaPipe Tasks API landmark indexing (0 to 20).
    """
    # Landmarks for finger tips and PIP joints (middle joint)
    # 4=Thumb tip, 8=Index tip, 12=Middle tip, 16=Ring tip, 20=Pinky tip
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # PIP joints (6=Index python, etc.)
    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    # Check if fingers are extended (tip is higher than pip joint - Note: y is 0 at top)
    fingers_open = {
        "index": index_tip.y < index_pip.y,
        "middle": middle_tip.y < middle_pip.y,
        "ring": ring_tip.y < ring_pip.y,
        "pinky": pinky_tip.y < pinky_pip.y
    }
    
    open_count = sum(fingers_open.values())
    
    if open_count == 0:
        return "Fist"
    elif open_count == 4:
        return "Open Hand"
    elif fingers_open["index"] and not fingers_open["middle"] and not fingers_open["ring"] and not fingers_open["pinky"]:
        return "Pointing"
    elif fingers_open["index"] and fingers_open["middle"] and not fingers_open["ring"] and not fingers_open["pinky"]:
        return "Peace Sign"
    
    return "Unknown"

def draw_landmarks(image, hand_landmarks_list):
    """ Draw simple points for the landmarks on the image. """
    h, w, _ = image.shape
    for landmarks in hand_landmarks_list:
        for landmark in landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

def main():
    # Setup Hand Landmarker with the Tasks API model
    # We downloaded the task file to hand_landmarker.task
    try:
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=get_result)
    except Exception as e:
        print("Error initializing HandLandmarker. Did you download 'hand_landmarker.task'?")
        print(e)
        return
        
    cap = cv2.VideoCapture(0)
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally for a mirror effect & convert to RGB for mediapipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # In LIVE_STREAM mode, we must provide an increasing timestamp in ms
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Send the image to the hand landmarker
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            display_image = frame.copy()
            gesture = "None"
            
            # Use the global populated by the async callback
            if latest_result is not None and latest_result.hand_landmarks:
                draw_landmarks(display_image, latest_result.hand_landmarks)
                
                # Evaluate gesture on the first detected hand
                gesture = detect_gesture(latest_result.hand_landmarks[0])
                
                h, w, _ = display_image.shape
                # Position text near the wrist (landmark 0)
                cx, cy = int(latest_result.hand_landmarks[0][0].x * w), int(latest_result.hand_landmarks[0][0].y * h)
                cv2.putText(display_image, gesture, (cx, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Global overlay text
            cv2.putText(display_image, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Show output
            cv2.imshow('Hand Gesture Recognition', display_image)
            
            # Press 'ESC' to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
