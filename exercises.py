from utils_math import calculate_angle, calculate_percentage, get_landmark_array

# MediaPipe Pose Landmark Indices
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
NOSE = 0

# Minimum visibility/confidence to trust a landmark
MIN_VISIBILITY = 0.6


def _is_visible(landmark):
    """Check if a landmark has enough confidence to be trusted."""
    return hasattr(landmark, 'visibility') and landmark.visibility > MIN_VISIBILITY


def _all_visible(landmarks, indices):
    """Check if all given landmark indices are visible enough."""
    return all(_is_visible(landmarks[i]) for i in indices)


class ExerciseEvaluator:
    def __init__(self):
        self.reps = 0
        self.state = "UP"

    def reset(self):
        self.reps = 0
        self.state = "UP"

    def evaluate(self, landmarks):
        """
        Returns: (percentage, color, status_text, posture, feedback_list)
        """
        raise NotImplementedError("Must implement evaluate")


# ================================
# HELPER: Compute torso lean angle
# ================================
def _torso_vertical_angle(shoulder, hip):
    """Angle of torso from vertical. 0 = perfectly upright."""
    import math
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = abs(math.degrees(math.atan2(dx, -dy)))
    return angle


# ================================
# STRENGTH EXERCISES
# ================================

class Squat(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_hip = get_landmark_array(landmarks[L_HIP])
        l_knee = get_landmark_array(landmarks[L_KNEE])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        r_hip = get_landmark_array(landmarks[R_HIP])
        r_knee = get_landmark_array(landmarks[R_KNEE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])

        l_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_angle + r_angle) / 2.0

        percentage = calculate_percentage(avg_knee_angle, min_angle=90, max_angle=170)

        # --- Posture detection ---
        if avg_knee_angle > 160:
            posture = "Standing upright"
        elif avg_knee_angle > 130:
            posture = "Quarter squat"
        elif avg_knee_angle > 110:
            posture = "Half squat"
        elif avg_knee_angle > 95:
            posture = "Parallel squat"
        else:
            posture = "Deep squat"

        # --- Form feedback (only when clearly squatting) ---
        feedback = []
        trusted = _all_visible(landmarks, [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER])

        if avg_knee_angle < 130 and trusted:
            # Torso lean — very generous threshold
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            torso_lean = _torso_vertical_angle(avg_shoulder, avg_hip)
            if torso_lean > 50:
                feedback.append("Keep your chest up, you are leaning forward too much")

            # Knee symmetry — only flag very large differences
            knee_diff = abs(l_angle - r_angle)
            if knee_diff > 30:
                feedback.append("Your knees are uneven, try to squat symmetrically")

            # Knees caving in — generous threshold
            l_hip_x, r_hip_x = l_hip[0], r_hip[0]
            hip_width = abs(l_hip_x - r_hip_x)
            knee_width = abs(l_knee[0] - r_knee[0])
            if hip_width > 0.05 and knee_width < hip_width * 0.5:
                feedback.append("Your knees are caving inward, push them out over your toes")

        if not feedback:
            feedback.append("Good form, keep it up")

        # State machine for reps
        if percentage >= 95 and self.state == "UP":
            self.state = "DOWN"
        if percentage <= 10 and self.state == "DOWN":
            self.state = "UP"
            self.reps += 1

        color = (0, 255, 0) if percentage > 90 else (0, 165, 255) if percentage > 50 else (0, 0, 255)
        status = f"Reps: {self.reps} | State: {self.state}"
        return percentage, color, status, posture, feedback


class PushUp(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        l_elbow = get_landmark_array(landmarks[L_ELBOW])
        l_wrist = get_landmark_array(landmarks[L_WRIST])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])
        r_elbow = get_landmark_array(landmarks[R_ELBOW])
        r_wrist = get_landmark_array(landmarks[R_WRIST])
        l_hip = get_landmark_array(landmarks[L_HIP])
        r_hip = get_landmark_array(landmarks[R_HIP])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])

        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2.0
        is_prone = (avg_hip_y - avg_shoulder_y) < 0.35

        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_angle + r_angle) / 2.0

        percentage = calculate_percentage(avg_elbow_angle, min_angle=90, max_angle=160)

        feedback = []

        if not is_prone:
            posture = "Standing - not in push-up position"
            return 0.0, (0, 0, 255), f"Reps: {self.reps} | Get in position", posture, ["Get into a plank position to start push-ups"]

        # --- Posture detection ---
        if avg_elbow_angle > 150:
            posture = "Plank position (arms extended)"
        elif avg_elbow_angle > 120:
            posture = "Lowering down"
        elif avg_elbow_angle > 100:
            posture = "Almost at the bottom"
        else:
            posture = "Bottom of push-up"

        # --- Form feedback ---
        trusted = _all_visible(landmarks, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE])

        if trusted:
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            avg_ankle = [(l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2]

            body_angle = calculate_angle(avg_shoulder, avg_hip, avg_ankle)
            if body_angle < 130:
                if avg_hip_y > avg_shoulder_y:
                    feedback.append("Your hips are sagging, tighten your core and lift your hips")
                else:
                    feedback.append("Your hips are piked too high, lower them to form a straight line")

            # Arm symmetry — only flag extreme differences
            arm_diff = abs(l_angle - r_angle)
            if arm_diff > 35:
                feedback.append("Your arms are uneven, push evenly with both hands")

        if not feedback:
            feedback.append("Good form, keep it up")

        if percentage >= 90 and self.state == "UP":
            self.state = "DOWN"
        if percentage <= 15 and self.state == "DOWN":
            self.state = "UP"
            self.reps += 1

        color = (255, 0, 0) if percentage > 85 else (0, 165, 255)
        status = f"Reps: {self.reps} | State: {self.state}"
        return percentage, color, status, posture, feedback


class BicepCurl(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        l_elbow = get_landmark_array(landmarks[L_ELBOW])
        l_wrist = get_landmark_array(landmarks[L_WRIST])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])
        r_elbow = get_landmark_array(landmarks[R_ELBOW])
        r_wrist = get_landmark_array(landmarks[R_WRIST])
        l_hip = get_landmark_array(landmarks[L_HIP])
        r_hip = get_landmark_array(landmarks[R_HIP])

        l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        active_angle = min(l_arm_angle, r_arm_angle)
        active_side = "left" if l_arm_angle < r_arm_angle else "right"

        percentage = calculate_percentage(active_angle, min_angle=40, max_angle=150)

        # --- Posture detection ---
        if active_angle > 140:
            posture = "Arms fully extended"
        elif active_angle > 110:
            posture = "Starting to curl"
        elif active_angle > 70:
            posture = "Mid curl"
        elif active_angle > 45:
            posture = "Near top of curl"
        else:
            posture = "Full curl at the top"

        # --- Form feedback ---
        feedback = []
        trusted = _all_visible(landmarks, [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP])

        if trusted:
            # Elbow drift — very generous (0.15 is a big drift)
            if active_side == "left":
                elbow_x, shoulder_x = l_elbow[0], l_shoulder[0]
                elbow_y, shoulder_y = l_elbow[1], l_shoulder[1]
            else:
                elbow_x, shoulder_x = r_elbow[0], r_shoulder[0]
                elbow_y, shoulder_y = r_elbow[1], r_shoulder[1]

            elbow_drift = abs(elbow_x - shoulder_x)
            if elbow_drift > 0.15:
                feedback.append("Keep your elbows close to your body, they are flaring out")

            # Shoulder rising — only flag extreme swing
            if elbow_y < shoulder_y - 0.12:
                feedback.append("Your elbow is rising too high, do not swing the weight")

            # Torso lean — very generous
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            torso_lean = _torso_vertical_angle(avg_shoulder, avg_hip)
            if torso_lean > 35:
                feedback.append("You are leaning back too much, keep your torso straight")

        if not feedback:
            feedback.append("Good form, keep it up")

        if percentage >= 90 and self.state == "DOWN":
            self.state = "UP"
        if percentage <= 10 and self.state == "UP":
            self.state = "DOWN"
            self.reps += 1

        color = (255, 255, 0) if percentage > 80 else (200, 200, 200)
        status = f"Reps: {self.reps} | State: {self.state}"
        return percentage, color, status, posture, feedback


class Lunge(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_hip = get_landmark_array(landmarks[L_HIP])
        l_knee = get_landmark_array(landmarks[L_KNEE])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        r_hip = get_landmark_array(landmarks[R_HIP])
        r_knee = get_landmark_array(landmarks[R_KNEE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])

        l_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)

        avg_knee = (l_angle + r_angle) / 2.0
        front_knee = min(l_angle, r_angle)
        back_knee = max(l_angle, r_angle)

        percentage = calculate_percentage(avg_knee, min_angle=95, max_angle=160)

        # --- Posture detection ---
        if avg_knee > 155:
            posture = "Standing upright"
        elif avg_knee > 130:
            posture = "Starting lunge descent"
        elif avg_knee > 110:
            posture = "Mid lunge"
        else:
            posture = "Deep lunge position"

        # --- Form feedback (only when clearly lunging) ---
        feedback = []
        trusted = _all_visible(landmarks, [L_HIP, R_HIP, L_KNEE, R_KNEE, L_SHOULDER, R_SHOULDER])

        if avg_knee < 140 and trusted:
            # Torso lean — generous
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            torso_lean = _torso_vertical_angle(avg_shoulder, avg_hip)
            if torso_lean > 40:
                feedback.append("Keep your torso upright, you are leaning forward")

            # Front knee — only flag extremes
            if front_knee < 65:
                feedback.append("Your front knee is bending too much, do not go past your toes")

            # Back leg — only if very straight while supposed to be lunging
            if back_knee > 165:
                feedback.append("Bend your back knee more, lower it towards the ground")

        if not feedback:
            feedback.append("Good form, keep it up")

        if percentage >= 90 and self.state == "UP":
            self.state = "DOWN"
        if percentage <= 15 and self.state == "DOWN":
            self.state = "UP"
            self.reps += 1

        color = (255, 0, 255) if percentage > 85 else (128, 0, 128)
        status = f"Reps: {self.reps} | State: {self.state}"
        return percentage, color, status, posture, feedback


# ================================
# YOGA POSES (Static)
# ================================

class TreePose(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_hip = get_landmark_array(landmarks[L_HIP])
        l_knee = get_landmark_array(landmarks[L_KNEE])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        r_hip = get_landmark_array(landmarks[R_HIP])
        r_knee = get_landmark_array(landmarks[R_KNEE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])
        l_wrist = get_landmark_array(landmarks[L_WRIST])
        r_wrist = get_landmark_array(landmarks[R_WRIST])

        l_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)

        straight_leg = max(l_angle, r_angle)
        bent_leg = min(l_angle, r_angle)

        straight_score = calculate_percentage(straight_leg, min_angle=170, max_angle=140)
        bent_score = calculate_percentage(bent_leg, min_angle=90, max_angle=150)

        percentage = (straight_score + bent_score) / 2.0

        # --- Posture detection ---
        if bent_leg > 140:
            posture = "Standing on both feet"
        elif bent_leg > 110:
            posture = "Lifting foot slightly"
        elif bent_leg > 80:
            posture = "Tree pose forming"
        else:
            posture = "Deep tree pose"

        # --- Form feedback ---
        feedback = []
        trusted = _all_visible(landmarks, [L_HIP, R_HIP, L_KNEE, R_KNEE, L_SHOULDER, R_SHOULDER])

        if trusted:
            # Standing leg — only if really bent
            if straight_leg < 145:
                feedback.append("Keep your standing leg straight, do not bend it")

            # Torso lean — generous
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            torso_lean = _torso_vertical_angle(avg_shoulder, avg_hip)
            if torso_lean > 25:
                feedback.append("Straighten your torso, you are leaning to the side")

        if not feedback:
            feedback.append("Beautiful tree pose, hold steady")

        color = (0, 255, 0) if percentage > 90 else (0, 200, 255)
        return percentage, color, "Hold Pose", posture, feedback


class WarriorIIPose(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_hip = get_landmark_array(landmarks[L_HIP])
        l_knee = get_landmark_array(landmarks[L_KNEE])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        r_hip = get_landmark_array(landmarks[R_HIP])
        r_knee = get_landmark_array(landmarks[R_KNEE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        l_elbow = get_landmark_array(landmarks[L_ELBOW])
        l_wrist = get_landmark_array(landmarks[L_WRIST])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])
        r_elbow = get_landmark_array(landmarks[R_ELBOW])
        r_wrist = get_landmark_array(landmarks[R_WRIST])

        l_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)

        straight_leg = max(l_angle, r_angle)
        bent_leg = min(l_angle, r_angle)
        leg_straight_score = calculate_percentage(straight_leg, min_angle=170, max_angle=140)
        leg_bent_score = calculate_percentage(bent_leg, min_angle=100, max_angle=160)

        l_arm = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm = calculate_angle(r_shoulder, r_elbow, r_wrist)
        arm_score = (calculate_percentage(l_arm, 170, 130) + calculate_percentage(r_arm, 170, 130)) / 2.0

        percentage = (leg_straight_score + leg_bent_score + arm_score) / 3.0

        # --- Posture detection ---
        if bent_leg > 150:
            posture = "Standing, not in warrior pose yet"
        elif bent_leg > 120:
            posture = "Starting warrior stance"
        elif bent_leg > 95:
            posture = "Warrior II pose"
        else:
            posture = "Deep warrior stance"

        # --- Form feedback ---
        feedback = []
        trusted = _all_visible(landmarks, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE])

        if trusted:
            # Back leg — only if very bent
            if straight_leg < 140:
                feedback.append("Straighten your back leg more")

            # Arms — only flag if really bent
            if l_arm < 140:
                feedback.append("Straighten your left arm more")
            if r_arm < 140:
                feedback.append("Straighten your right arm more")

            # Arms horizontal — very generous (0.18 is a big gap)
            avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            if abs(l_wrist[1] - avg_shoulder_y) > 0.18:
                feedback.append("Raise your left arm closer to shoulder height")
            if abs(r_wrist[1] - avg_shoulder_y) > 0.18:
                feedback.append("Raise your right arm closer to shoulder height")

            # Torso lean — generous
            avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
            avg_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
            torso_lean = _torso_vertical_angle(avg_shoulder, avg_hip)
            if torso_lean > 35:
                feedback.append("Keep your torso upright, do not lean forward")

        if not feedback:
            feedback.append("Excellent warrior pose, hold it strong")

        color = (255, 165, 0) if percentage > 85 else (100, 100, 100)
        return percentage, color, "Hold Pose", posture, feedback


class DownwardDogPose(ExerciseEvaluator):
    def evaluate(self, landmarks):
        l_shoulder = get_landmark_array(landmarks[L_SHOULDER])
        l_hip = get_landmark_array(landmarks[L_HIP])
        l_knee = get_landmark_array(landmarks[L_KNEE])
        l_ankle = get_landmark_array(landmarks[L_ANKLE])
        l_elbow = get_landmark_array(landmarks[L_ELBOW])
        l_wrist = get_landmark_array(landmarks[L_WRIST])
        r_shoulder = get_landmark_array(landmarks[R_SHOULDER])
        r_hip = get_landmark_array(landmarks[R_HIP])
        r_knee = get_landmark_array(landmarks[R_KNEE])
        r_ankle = get_landmark_array(landmarks[R_ANKLE])
        r_elbow = get_landmark_array(landmarks[R_ELBOW])
        r_wrist = get_landmark_array(landmarks[R_WRIST])

        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0

        l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_arm_angle = (l_arm_angle + r_arm_angle) / 2.0

        hip_score = calculate_percentage(avg_hip_angle, min_angle=90, max_angle=160)
        knee_score = calculate_percentage(avg_knee_angle, min_angle=170, max_angle=130)

        percentage = (hip_score + knee_score) / 2.0

        # --- Posture detection ---
        if avg_hip_angle > 140:
            posture = "Standing or plank, not in downward dog"
        elif avg_hip_angle > 110:
            posture = "Forming inverted V shape"
        else:
            posture = "Downward dog position"

        # --- Form feedback ---
        feedback = []
        trusted = _all_visible(landmarks, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE])

        if trusted:
            # Hip angle — only flag extremes
            if avg_hip_angle > 135:
                feedback.append("Push your hips higher towards the ceiling")
            elif avg_hip_angle < 60:
                feedback.append("Your hips are too high, lower them slightly")

            # Legs straight — generous
            if avg_knee_angle < 140:
                feedback.append("Straighten your legs, press your heels towards the floor")

            # Arms straight — generous
            if avg_arm_angle < 140:
                feedback.append("Straighten your arms fully, push the ground away")

        if not feedback:
            feedback.append("Great downward dog, breathe and hold")

        color = (0, 200, 200) if percentage > 85 else (50, 50, 50)
        return percentage, color, "Hold Pose", posture, feedback


# Helper dictionary to fetch by name
EXERCISE_MODELS = {
    "Squat": Squat(),
    "Push-Up": PushUp(),
    "Bicep Curl": BicepCurl(),
    "Lunge": Lunge(),
    "Tree Pose": TreePose(),
    "Warrior II": WarriorIIPose(),
    "Downward Dog": DownwardDogPose()
}
