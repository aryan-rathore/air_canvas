import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
canvas = None
prev_x, prev_y = None, None
def draw_on_canvas(image, x, y, prev_x, prev_y, color=(0, 0, 255), thickness=5):
    if prev_x is not None and prev_y is not None:
        cv2.line(image, (prev_x, prev_y), (x, y), color, thickness)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Use the tip of the index finger for drawing
                if id == 8:
                    if prev_x is None and prev_y is None:
                        prev_x, prev_y = cx, cy
                    else:
                        draw_on_canvas(canvas, cx, cy, prev_x, prev_y)
                        prev_x, prev_y = cx, cy

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame = cv2.add(frame, canvas)
    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Clear the canvas when 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
