import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

dataset_path = r"D:\AI_Projects\archive (5)\leapGestRecog"

data = []

for person_folder in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_folder)

    if not os.path.isdir(person_path):
        continue

    for gesture_folder in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        label = gesture_folder

        for image_name in os.listdir(gesture_path):

            image_path = os.path.join(gesture_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                landmarks = []

                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks.append(label)
                data.append(landmarks)

df = pd.DataFrame(data)
df.to_csv("gesture_data.csv", index=False)

print("CSV created successfully!")
print("Total samples:", len(data))