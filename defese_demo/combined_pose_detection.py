import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_pose_choice():
    print("Select a pose:")
    print("1. Plank")
    print("2. Warrior II")
    print("3. Mountain")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ").strip()
    if choice == '1':
        return 'plank'
    elif choice == '2':
        return 'warrior2'
    elif choice == '3':
        return 'mountain'
    elif choice == '4':
        return None
    else:
        print("Invalid choice. Please try again.")
        return get_pose_choice()

def set_configurations(pose):
    if pose == 'plank':
        IMPORTANT_LMS = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX",
        ]
        video_list = ['plank test.mp4','Phalakasana-9.mp4']
        sklearn_class_map = {0: "C", 1: "H", 2: "L"}
        threshold_sklearn = 0.82
        stage_map_sklearn = {"C": "Correct", "H": "High back", "L": "Low back"}
    elif pose == 'mountain':
        IMPORTANT_LMS = [
            "NOSE",
            "LEFT_EAR",
            "RIGHT_EAR",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_INDEX",
            "RIGHT_INDEX",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
        ]
        video_list = ['sample1.mp4','sample1.mp4']
        sklearn_class_map = {0:'c', 1:'feet_too_close', 2:'arms_dropped', 3:'shoulders_raised', 4:'leaning_forward', 5:'leaning_back', 6:'head_tilted'}
        threshold_sklearn = 0.80
        stage_map_sklearn = {'c': "Correct", 'feet_too_close': "Feet Too Close", 'arms_dropped': "Arms Dropped", 'shoulders_raised': "Shoulders Raised", 'leaning_forward': "Leaning Forward", 'leaning_back': "Leaning Back", 'head_tilted': "Head Tilted"}
    elif pose == 'warrior2':
        IMPORTANT_LMS = [
            "NOSE",
            "LEFT_EYE",
            "RIGHT_EYE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_INDEX",
            "RIGHT_INDEX",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX",
        ]
        video_list = ['warrior-ii -1.mp4','Virabhadrasana ii -3.mp4']
        sklearn_class_map = {0:'c', 1:'bent_front_knee', 2:'arms_dropped', 3:'arms_bent', 4:'hips_rotated', 5:'back_foot_wrong_angle', 6:'leaning_forward', 7:'narrow_stance', 8:'shoulders_not_aligned'}
        threshold_sklearn = 0.80
        stage_map_sklearn = {'c': "Correct", 'bent_front_knee': "Bent Front Knee", 'arms_dropped': "Arms Dropped", 'arms_bent': "Arms Bent", 'hips_rotated': "Hips Rotated", 'back_foot_wrong_angle': "Back Foot Wrong Angle", 'leaning_forward': "Leaning Forward", 'narrow_stance': "Narrow Stance", 'shoulders_not_aligned': "Shoulders Not Aligned"}
    else:
        raise ValueError("Invalid pose selected")

    # Generate headers
    HEADERS = ["label"]
    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    return IMPORTANT_LMS, HEADERS, video_list, sklearn_class_map, threshold_sklearn, stage_map_sklearn

def select_video(video_list, pose):
    selected_video = random.choice(video_list)
    video_path = f'./{pose}/{selected_video}'
    print(f"Selected video: {video_path}")
    return video_path

def load_models(pose):
    with open(f"{pose}_model.pkl", "rb") as f:
        sklearn_model = pickle.load(f)
    with open(f"{pose}_input_scaler.pkl", "rb") as f2:
        input_scaler = pickle.load(f2)
    return sklearn_model, input_scaler

def extract_important_keypoints(results, IMPORTANT_LMS):
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def run_detection(video_path, IMPORTANT_LMS, HEADERS, sklearn_model, input_scaler, sklearn_class_map, threshold_sklearn, stage_map_sklearn):
    def get_class(prediction):
        return sklearn_class_map.get(prediction)

    cap = cv2.VideoCapture(video_path)
    current_stage = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Reduce size of a frame
            image = rescale_frame(image, 100)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                print("No human found")
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Make detection
            try:
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                predicted_class_num = sklearn_model.predict(X)[0]
                predicted_class = get_class(predicted_class_num)
                prediction_probability = sklearn_model.predict_proba(X)[0]

                # Evaluate model prediction
                if prediction_probability[predicted_class_num] >= threshold_sklearn:
                    current_stage = stage_map_sklearn[predicted_class]
                else:
                    current_stage = "Unknown"
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (400, 70), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "CLASS", (130, 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (125, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
 
                # Display probability
                cv2.putText(image, "PROB", (20, 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prediction_probability[predicted_class_num], 2)), (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
            
            cv2.namedWindow("CV2", cv2.WINDOW_NORMAL)
            cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

def main():
    while True:
        pose = get_pose_choice()
        if pose is None:
            break
        print(f"Selected pose: {pose}")
        IMPORTANT_LMS, HEADERS, video_list, sklearn_class_map, threshold_sklearn, stage_map_sklearn = set_configurations(pose)
        video_path = select_video(video_list, pose)
        sklearn_model, input_scaler = load_models(pose)
        run_detection(video_path, IMPORTANT_LMS, HEADERS, sklearn_model, input_scaler, sklearn_class_map, threshold_sklearn, stage_map_sklearn)

if __name__ == "__main__":
    main()