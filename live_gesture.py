import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("gesture_model.pkl")

gesture_names = {
    "01_palm": "Open Palm",
    "02_l": "L Sign",
    "03_fist": "Closed Fist",
    "04_fist_moved": "Fist Side",
    "05_thumb": "Thumbs Up",
    "06_index": "Pointing Finger",
    "07_ok": "OK Sign",
    "08_palm_moved": "Palm Side",
    "09_c": "C Sign",
    "10_down": "Thumbs Down"
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            features = []

            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([features])[0]
            prediction = gesture_names.get(prediction, prediction)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                str(prediction),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()