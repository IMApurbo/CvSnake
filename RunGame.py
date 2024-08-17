import cv2
import mediapipe as mp
import numpy as np
import random
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Snake Game Variables
snake_points = []
snake_length = 150
current_length = 0
score = 0
food_pos = [random.randint(100, 1000), random.randint(100, 600)]
food_radius = 20
game_over = False

# Function to detect collision with food
def check_food_collision(head_pos, food_pos, food_radius):
    distance = math.hypot(head_pos[0] - food_pos[0], head_pos[1] - food_pos[1])
    if distance < food_radius + 20:  # 20 is the snake's head radius
        return True
    return False

# Function to reset game
def reset_game():
    global snake_points, current_length, score, snake_length, food_pos, game_over
    snake_points = []
    current_length = 0
    score = 0
    snake_length = 150
    food_pos = [random.randint(100, 1000), random.randint(100, 600)]
    game_over = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                lmList.append([int(lm.x * w), int(lm.y * h)])
                
            # Get the index finger tip position
            finger_tip = lmList[8]
            cv2.circle(img, (finger_tip[0], finger_tip[1]), 10, (255, 0, 0), cv2.FILLED)
            
            if not game_over:
                if snake_points:
                    prev_point = snake_points[-1]
                    dist = math.hypot(finger_tip[0] - prev_point[0], finger_tip[1] - prev_point[1])
                    if dist > 20:  # Minimum distance to move
                        snake_points.append(finger_tip)
                        current_length += dist
                else:
                    snake_points.append(finger_tip)
                
                # Limit the length of the snake
                while current_length > snake_length:
                    diff = math.hypot(snake_points[1][0] - snake_points[0][0], snake_points[1][1] - snake_points[0][1])
                    current_length -= diff
                    snake_points.pop(0)
                
                # Check for collision with food
                if check_food_collision(snake_points[-1], food_pos, food_radius):
                    score += 1
                    snake_length += 30
                    food_pos = [random.randint(100, 1000), random.randint(100, 600)]
                
                # Draw the snake
                for i in range(len(snake_points) - 1):
                    cv2.line(img, tuple(snake_points[i]), tuple(snake_points[i + 1]), (0, 255, 0), 20)
                cv2.circle(img, tuple(snake_points[-1]), 15, (0, 0, 255), cv2.FILLED)
                
                # Draw food
                cv2.circle(img, tuple(food_pos), food_radius, (0, 255, 255), cv2.FILLED)
                
                # Check for self-collision
                if len(snake_points) > 4:
                    pts = np.array(snake_points[:-2], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, (255, 0, 0), 3)
                    minDist = cv2.pointPolygonTest(pts, tuple(snake_points[-1]), True)
                    if minDist >= -1 and minDist <= 1:
                        game_over = True
            
            # Draw game over
            if game_over:
                cv2.putText(img, 'Game Over', (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(img, f'Score: {score}', (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Snake Game", img)
    
    # Reset game if 'r' is pressed
    key = cv2.waitKey(1)
    if key == ord('r'):
        reset_game()
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
