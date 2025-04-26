import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import win32api
import win32con
import time
from math import sqrt, hypot
from collections import deque

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize webcam
video = cv2.VideoCapture(0)

# Mouse control states
click_state = False  # for drag functionality
scroll_mode = False
zoom_mode = False
last_gesture = None
gesture_start_time = 0
gesture_cooldown = 0.5  # seconds

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Movement smoothing
smoothing = 5
position_history = deque(maxlen=smoothing)
scroll_accumulator = 0

# Gesture history for recognition
gesture_history = []
gesture_history_max = 10

# Status display
status_text = "Ready"
status_color = (0, 255, 0)

# Define gestures
def get_landmark_position(landmarks, index, width, height):
    """Get the position of a specific landmark"""
    lm = landmarks.landmark[index]
    return int(lm.x * width), int(lm.y * height)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return hypot(point1[0] - point2[0], point1[1] - point2[1])

def is_finger_raised(landmarks, finger_tip_idx, finger_pip_idx, width, height):
    """Check if a finger is raised by comparing y coordinates"""
    tip_y = landmarks.landmark[finger_tip_idx].y * height
    pip_y = landmarks.landmark[finger_pip_idx].y * height
    return tip_y < pip_y

def recognize_gesture(landmarks, width, height):
    """Recognize hand gestures based on finger positions"""
    # Get finger tip positions
    thumb_tip = get_landmark_position(landmarks, mp_hands.HandLandmark.THUMB_TIP, width, height)
    index_tip = get_landmark_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, width, height)
    middle_tip = get_landmark_position(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, width, height)
    ring_tip = get_landmark_position(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, width, height)
    pinky_tip = get_landmark_position(landmarks, mp_hands.HandLandmark.PINKY_TIP, width, height)
    
    # Check if fingers are raised
    index_raised = is_finger_raised(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                  mp_hands.HandLandmark.INDEX_FINGER_PIP, width, height)
    middle_raised = is_finger_raised(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, width, height)
    ring_raised = is_finger_raised(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, 
                                 mp_hands.HandLandmark.RING_FINGER_PIP, width, height)
    pinky_raised = is_finger_raised(landmarks, mp_hands.HandLandmark.PINKY_TIP, 
                                  mp_hands.HandLandmark.PINKY_TIP, width, height)
    
    # Calculate distances
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    
    # Recognize gestures
    if index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "POINTER", index_tip  # Just pointing (mouse move)
    
    elif index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "RIGHT_CLICK", index_tip  # Index and middle finger up (right click)
    
    elif thumb_index_dist < 40:  # Pinch gesture
        return "LEFT_CLICK", index_tip
    
    elif index_raised and middle_raised and ring_raised and not pinky_raised:
        return "SCROLL_MODE", index_tip  # Three fingers up (scroll mode)
    
    elif index_raised and middle_raised and ring_raised and pinky_raised:
        return "SCREENSHOT", index_tip  # All fingers up (screenshot)
    
    elif not index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "FIST", index_tip  # Fist (pause)
    
    elif not index_raised and not middle_raised and not ring_raised and pinky_raised:
        return "ROCK_N_ROLL", index_tip  # Rock and roll sign (quick launch)
    
    return "UNKNOWN", index_tip

def apply_smoothing(point):
    """Apply smoothing to cursor movement"""
    position_history.append(point)
    avg_x = sum(p[0] for p in position_history) // len(position_history)
    avg_y = sum(p[1] for p in position_history) // len(position_history)
    return (avg_x, avg_y)

def draw_status(frame, text, color=(0, 255, 0)):
    """Draw status text on frame"""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Main loop
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            print("Failed to read from webcam")
            continue
        
        # Flip the frame horizontally for a more natural feeling
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert the BGR image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(rgb)
        
        # Convert back to BGR for display
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        current_time = time.time()
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Recognize gesture
                gesture, cursor_point = recognize_gesture(hand_landmarks, w, h)
                
                # Map cursor point to screen coordinates with smoothing
                screen_x = np.interp(cursor_point[0], [100, w-100], [0, screen_w])
                screen_y = np.interp(cursor_point[1], [100, h-100], [0, screen_h])
                
                smoothed_x, smoothed_y = apply_smoothing((int(screen_x), int(screen_y)))
                
                # Handle gestures with cooldown to prevent rapid triggering
                if last_gesture != gesture or (current_time - gesture_start_time) > gesture_cooldown:
                    last_gesture = gesture
                    gesture_start_time = current_time
                    
                    if gesture == "POINTER":
                        # Just move the cursor
                        win32api.SetCursorPos((smoothed_x, smoothed_y))
                        status_text = "Pointer Mode"
                        status_color = (0, 255, 0)
                        if click_state:
                            pyautogui.mouseUp()
                            click_state = False
                        if scroll_mode:
                            scroll_mode = False
                    
                    elif gesture == "LEFT_CLICK" and not click_state:
                        # Left click
                        win32api.SetCursorPos((smoothed_x, smoothed_y))
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, smoothed_x, smoothed_y, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, smoothed_x, smoothed_y, 0, 0)
                        status_text = "Left Click"
                        status_color = (255, 0, 0)
                        time.sleep(0.2)  # Debounce
                    
                    elif gesture == "RIGHT_CLICK":
                        # Right click
                        win32api.SetCursorPos((smoothed_x, smoothed_y))
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, smoothed_x, smoothed_y, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, smoothed_x, smoothed_y, 0, 0)
                        status_text = "Right Click"
                        status_color = (0, 0, 255)
                        time.sleep(0.3)  # Debounce
                    
                    elif gesture == "SCROLL_MODE":
                        # Toggle scroll mode
                        scroll_mode = not scroll_mode
                        status_text = "Scroll Mode: " + ("ON" if scroll_mode else "OFF")
                        status_color = (255, 255, 0)
                    
                    elif gesture == "FIST":
                        # Pause - do nothing but show status
                        status_text = "Paused"
                        status_color = (128, 128, 128)
                    
                    elif gesture == "SCREENSHOT":
                        # Take screenshot
                        screenshot_path = f"screenshot_{int(time.time())}.png"
                        pyautogui.screenshot(screenshot_path)
                        status_text = f"Screenshot saved: {screenshot_path}"
                        status_color = (255, 165, 0)
                        time.sleep(1)  # Longer cooldown for screenshot
                    
                    elif gesture == "ROCK_N_ROLL":
                        # Quick launch action (open calculator as an example)
                        pyautogui.hotkey('win', 'r')
                        time.sleep(0.5)
                        pyautogui.typewrite('calc\n')
                        status_text = "Launching Calculator"
                        status_color = (255, 0, 255)
                        time.sleep(1)  # Longer cooldown for app launch
                
                # Continue handling active modes
                if scroll_mode:
                    # Use vertical position for scrolling
                    current_y = smoothed_y
                    scroll_amount = int((screen_h/2 - current_y) / 20)
                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)
                    status_text = f"Scrolling: {scroll_amount}"
                    status_color = (255, 255, 0)
                
                # Move the cursor for all gestures except when paused
                if gesture != "FIST":
                    win32api.SetCursorPos((smoothed_x, smoothed_y))
        
        else:
            # No hands detected
            status_text = "No hands detected"
            status_color = (128, 128, 128)
            
            # Release any active mouse buttons if hands disappear
            if click_state:
                pyautogui.mouseUp()
                click_state = False
        
        # Draw status on frame
        draw_status(frame, status_text, status_color)
        
        # Display control instructions
        cv2.putText(frame, "Controls:", (10, h-160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Index finger: Move cursor", (10, h-130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Pinch (thumb+index): Click", (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Index+Middle: Right click", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Three fingers: Scroll mode", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Press 'x' to exit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Hand Gesture Controls', frame)
        
        # Exit on 'x' press
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

# Clean up
video.release()
cv2.destroyAllWindows()