import cv2
import mediapipe as mp
import math
import time
import winsound



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
last_play = 0

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # draw 6 strings
    for i in range(1, 6):
        cv2.line(frame, (0, i*h//6), (w, i*h//6), (255, 0, 0), 1)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        thumb = hand.landmark[4]
        index = hand.landmark[8]

        d = distance(thumb, index)
        y = int(index.y * h)
        string_no = min(5, y // (h // 6))

        if d < 0.04 and time.time() - last_play > 0.4:
            sound_path = rf"f:\5TH SEM\air_guitar\string{string_no+1}.wav"
            winsound.PlaySound(sound_path, winsound.SND_ASYNC)
            last_play = time.time()

            cv2.putText(frame, f"STRING {string_no+1}",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

    cv2.imshow("Air Guitar", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
