import cv2
import mediapipe as mp
import math
import pygame

# --------- Audio (supports overlapping sounds) ---------
pygame.mixer.init()

sounds = [
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string1.wav"),
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string2.wav"),
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string3.wav"),
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string4.wav"),
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string5.wav"),
    pygame.mixer.Sound(r"f:\5TH SEM\air_guitar\string6.wav"),
]

# --------- MediaPipe ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# --------- Camera ---------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not opened")
    exit()

# --------- Variables ---------
prev_string = None
prev_y = None

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# --------- Main Loop ---------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw 6 guitar strings
    for i in range(1, 6):
        cv2.line(frame, (0, i*h//6), (w, i*h//6), (255, 0, 0), 1)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        thumb = hand.landmark[4]
        index = hand.landmark[8]

        # Pinch detection (pluck enable)
        if distance(thumb, index) < 0.05:
            y = int(index.y * h)
            current_string = min(5, y // (h // 6))

            # Play sound when crossing strings (continuous strum)
            if prev_string is not None and current_string != prev_string:
                sounds[current_string].play()  # OVERLAPS perfectly

                if prev_y is not None:
                    direction = "DOWN STRUM" if y > prev_y else "UP STRUM"
                else:
                    direction = "STRUM"

                cv2.putText(
                    frame,
                    f"{direction} | STRING {current_string+1}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 0),
                    3
                )

            prev_string = current_string
            prev_y = y
        else:
            prev_string = None
            prev_y = None

    cv2.imshow("Air Guitar - Continuous Strumming", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
