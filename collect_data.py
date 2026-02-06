import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data_file = "dataset/data.csv"

if not os.path.exists("dataset"):
    os.makedirs("dataset")

with open(data_file, "a", newline="") as f:
    writer = csv.writer(f)

    print("Press keys to save gesture:")
    print("0 = Palm | 1 = Fist | 2 = Thumbs Up | 3 = Peace")

    while True:
        ret, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        cv2.imshow("Collect Data", img)

        key = cv2.waitKey(1)

        if key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            if results.multi_hand_landmarks:
                label = key - 48
                writer.writerow(landmarks + [label])
                print(f"Saved gesture {label}")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
