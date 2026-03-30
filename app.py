import cv2
import time
import threading
import subprocess
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe Tasks API not found. Please ensure mediapipe is installed correctly.")
    exit()

from exercises import EXERCISE_MODELS

# List of available exercises
EXERCISE_NAMES = list(EXERCISE_MODELS.keys())
current_idx = 0
current_exercise_name = EXERCISE_NAMES[current_idx]

# Global variables for async callback results
latest_result = None


# ================================
# Text-to-Speech via Windows SAPI (PowerShell) — never freezes
# ================================
class SpeechEngine:
    def __init__(self):
        self._last_spoken = ""
        self._last_speak_time = 0
        self._cooldown = 3.0
        self._lock = threading.Lock()
        self._speaking = False  # True while a subprocess is running

    def speak(self, text, force=False):
        """Speak text via PowerShell in background. Skips if already speaking or on cooldown."""
        if not text:
            return
        now = time.time()
        with self._lock:
            if not force and text == self._last_spoken and (now - self._last_speak_time) < self._cooldown:
                return
            if self._speaking:
                return
            self._last_spoken = text
            self._last_speak_time = now
            self._speaking = True

        t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
        t.start()

    def _do_speak(self, text):
        try:
            # Escape single quotes for PowerShell
            safe_text = text.replace("'", "''")
            cmd = (
                f"powershell -NoProfile -Command \""
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 2; "
                f"$s.Speak('{safe_text}'); "
                f"$s.Dispose()\""
            )
            subprocess.run(cmd, shell=True, timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        finally:
            with self._lock:
                self._speaking = False


speech = SpeechEngine()


def get_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw_progress_bar(img, percentage, color, x=50, y=100, w=30, h=300):
    """Draw a vertical progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 3)
    fill_h = int((percentage / 100.0) * h)
    cv2.rectangle(img, (x, y + h - fill_h), (x + w, y + h), color, cv2.FILLED)
    cv2.putText(img, f"{int(percentage)}%", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_landmarks(image, pose_landmarks):
    """Draw points and connecting lines for the pose landmarks."""
    h, w, _ = image.shape
    points = {}
    for idx, landmark in enumerate(pose_landmarks):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        points[idx] = (cx, cy)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
        (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
        (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
        (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in points and end_idx in points:
            cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)


def draw_feedback_panel(image, posture, feedback_list):
    """Draw a compact posture/feedback box in the top-right corner."""
    h, w, _ = image.shape

    # Calculate panel height based on content
    max_chars = 30
    total_lines = 3  # header lines (posture label + value + feedback header)
    for fb in feedback_list[:4]:
        total_lines += 1 + (len(fb) // max_chars)  # wrapped lines
    panel_h = 50 + total_lines * 25
    panel_h = min(panel_h, 350)  # cap height

    panel_w = 360
    panel_x = w - panel_w - 10  # 10px margin from right edge
    panel_y = 10  # 10px margin from top

    # Semi-transparent dark box (only covers the small panel area)
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    # Border
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (100, 100, 100), 1)

    # Posture label
    tx = panel_x + 12
    ty = panel_y + 25
    cv2.putText(image, "POSTURE:", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, posture, (tx + 90, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Divider
    ty += 12
    cv2.line(image, (tx, ty), (panel_x + panel_w - 12, ty), (80, 80, 80), 1)

    # Feedback header
    ty += 20
    cv2.putText(image, "FEEDBACK:", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
    ty += 25

    # Feedback items with word wrapping
    for fb in feedback_list[:4]:
        if ty > panel_y + panel_h - 15:
            break
        if "good" in fb.lower() or "great" in fb.lower() or "excellent" in fb.lower() or "beautiful" in fb.lower():
            fb_color = (0, 255, 100)
            bullet = "OK"
        else:
            fb_color = (0, 140, 255)
            bullet = "!!"

        words = fb.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for j, line in enumerate(lines):
            if ty > panel_y + panel_h - 15:
                break
            prefix = f"[{bullet}] " if j == 0 else "     "
            cv2.putText(image, prefix + line, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, fb_color, 1, cv2.LINE_AA)
            ty += 22
        ty += 5


def main():
    global current_idx, current_exercise_name

    # Setup Pose Landmarker
    try:
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            result_callback=get_result)
    except Exception as e:
        print("Error initializing Pose Landmarker. Missing 'pose_landmarker_full.task'?")
        print(e)
        return

    # Try external cameras first (index 1, 2, ...), fall back to default (0)
    cap = None
    for cam_idx in range(1, 5):
        test_cap = cv2.VideoCapture(cam_idx)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"External camera detected at index {cam_idx}, using it.")
                break
            test_cap.release()
    if cap is None:
        cap = cv2.VideoCapture(0)
        print("Using default camera (index 0).")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Starting Tracker. Press 'N' to switch exercise, 'ESC' to exit.")
        print("Real-time posture detection and voice feedback enabled.")

        cv2.namedWindow('AI Fitness & Yoga Tracker', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('AI Fitness & Yoga Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        prev_reps = {}  # track reps per exercise to detect new rep

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            display_image = frame.copy()

            percentage = 0.0
            color = (0, 255, 0)
            status_text = "Waiting for Pose..."
            posture = "No pose detected"
            feedback_list = []

            if latest_result is not None and latest_result.pose_landmarks:
                landmarks = latest_result.pose_landmarks[0]
                draw_landmarks(display_image, landmarks)

                evaluator = EXERCISE_MODELS[current_exercise_name]
                result = evaluator.evaluate(landmarks)
                percentage, color, status_text, posture, feedback_list = result

                # Check if a new rep was completed
                current_reps = evaluator.reps
                old_reps = prev_reps.get(current_exercise_name, 0)

                if current_reps > old_reps:
                    prev_reps[current_exercise_name] = current_reps
                    # Build rep announcement with form assessment
                    has_error = any(
                        "good" not in fb.lower() and "great" not in fb.lower()
                        and "excellent" not in fb.lower() and "beautiful" not in fb.lower()
                        for fb in feedback_list
                    )
                    form_status = "form is incorrect" if has_error else "good form"
                    # Find the specific correction if any
                    correction = ""
                    for fb in feedback_list:
                        if "good" not in fb.lower() and "great" not in fb.lower() and "excellent" not in fb.lower() and "beautiful" not in fb.lower():
                            correction = fb
                            break
                    rep_word = "rep" if current_reps == 1 else "reps"
                    rep_announcement = f"{current_exercise_name}, {current_reps} {rep_word}, {form_status}"
                    if correction:
                        rep_announcement += f". {correction}"
                    speech.speak(rep_announcement, force=True)
                else:
                    # Between reps, speak form corrections normally
                    speech_text = ""
                    for fb in feedback_list:
                        if "good" not in fb.lower() and "great" not in fb.lower() and "excellent" not in fb.lower() and "beautiful" not in fb.lower():
                            speech_text = fb
                            break
                    if speech_text:
                        speech.speak(speech_text)

            # Draw UI
            # 1. Exercise Title Bar
            cv2.rectangle(display_image, (0, 0), (640, 60), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_image, f"Mode: {current_exercise_name}", (20, 35),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_image, "(Press 'N' to switch)", (400, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 2. Status (Reps / Hold)
            cv2.putText(display_image, status_text, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 3. Progress Bar
            draw_progress_bar(display_image, percentage, color, x=30, y=120)

            # 4. Posture & Feedback Panel
            draw_feedback_panel(display_image, posture, feedback_list)

            cv2.imshow('AI Fitness & Yoga Tracker', display_image)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('n') or key == ord('N'):
                current_idx = (current_idx + 1) % len(EXERCISE_NAMES)
                current_exercise_name = EXERCISE_NAMES[current_idx]
                EXERCISE_MODELS[current_exercise_name].reset()
                speech.speak(f"Switched to {current_exercise_name}", force=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
