import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize webcam
video = cv2.VideoCapture(0)

# Canvas settings
canvas_width, canvas_height = 1280, 720
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
drawing_background = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Black background for drawing

# Drawing settings
drawing_color = (0, 255, 255)  # Yellow by default
line_thickness = 5
is_drawing = False
last_point = None

# Tool selection
tools = ["pen", "eraser", "color_picker", "clear"]
current_tool = "pen"

# Color palette
colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (255, 255, 255) # White
]
current_color_index = 0

# UI elements
ui_margin = 10
color_box_size = 30
tool_box_size = 50
color_palette_y = ui_margin
tools_y = color_palette_y + color_box_size + ui_margin

# Movement smoothing
smoothing = 5
position_history = deque(maxlen=smoothing)

# Status display
status_text = "Ready"
status_color = (0, 255, 0)

# Gesture history for stability
gesture_cooldown = 0.5  # seconds
last_gesture = None
gesture_start_time = 0

def get_landmark_position(landmarks, index, width, height):
    """Get the position of a specific landmark"""
    lm = landmarks.landmark[index]
    return int(lm.x * width), int(lm.y * height)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)

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
    
    # Get pip (second joint) positions for comparison
    index_pip = get_landmark_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_PIP, width, height)
    middle_pip = get_landmark_position(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, width, height)
    ring_pip = get_landmark_position(landmarks, mp_hands.HandLandmark.RING_FINGER_PIP, width, height)
    pinky_pip = get_landmark_position(landmarks, mp_hands.HandLandmark.PINKY_TIP, width, height)
    
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
    
    # Recognize gestures
    if index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "HOVER", index_tip  # Just pointing (hover)
    
    elif thumb_index_dist < 40 and index_raised:  # Pinch gesture
        return "DRAW", index_tip  # Draw with index finger
    
    elif index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "ERASER", index_tip  # Index and middle finger up (eraser)
    
    elif index_raised and middle_raised and ring_raised and not pinky_raised:
        return "COLOR_PICKER", index_tip  # Three fingers up (color picker)
    
    elif index_raised and middle_raised and ring_raised and pinky_raised:
        return "CLEAR", index_tip  # All fingers up (clear canvas)
    
    return "HOVER", index_tip  # Default to hover mode

def apply_smoothing(point):
    """Apply smoothing to cursor movement"""
    position_history.append(point)
    avg_x = sum(p[0] for p in position_history) // len(position_history)
    avg_y = sum(p[1] for p in position_history) // len(position_history)
    return (avg_x, avg_y)

def draw_status(frame, text, color=(0, 255, 0)):
    """Draw status text on frame"""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_ui(frame):
    """Draw UI elements on frame"""
    # Draw color palette
    for i, color in enumerate(colors):
        x = ui_margin + i * (color_box_size + ui_margin)
        y = color_palette_y
        cv2.rectangle(frame, (x, y), (x + color_box_size, y + color_box_size), color, -1)
        if i == current_color_index:
            cv2.rectangle(frame, (x-2, y-2), (x + color_box_size+2, y + color_box_size+2), (255, 255, 255), 2)
    
    # Draw tool icons
    for i, tool in enumerate(tools):
        x = ui_margin + i * (tool_box_size + ui_margin)
        y = tools_y
        cv2.rectangle(frame, (x, y), (x + tool_box_size, y + tool_box_size), (100, 100, 100), -1)
        
        # Highlight current tool
        if tool == current_tool:
            cv2.rectangle(frame, (x-2, y-2), (x + tool_box_size+2, y + tool_box_size+2), (255, 255, 255), 2)
        
        # Draw tool icon
        icon_center = (x + tool_box_size//2, y + tool_box_size//2)
        if tool == "pen":
            cv2.circle(frame, icon_center, 5, drawing_color, -1)
        elif tool == "eraser":
            cv2.rectangle(frame, (icon_center[0]-10, icon_center[1]-10), 
                         (icon_center[0]+10, icon_center[1]+10), (255, 255, 255), -1)
            cv2.line(frame, (icon_center[0]-5, icon_center[1]-5), 
                    (icon_center[0]+5, icon_center[1]+5), (100, 100, 100), 2)
        elif tool == "color_picker":
            cv2.circle(frame, icon_center, 10, (0, 0, 255), -1)
            cv2.circle(frame, icon_center, 7, (0, 255, 0), -1)
            cv2.circle(frame, icon_center, 4, (255, 0, 0), -1)
        elif tool == "clear":
            cv2.line(frame, (icon_center[0]-10, icon_center[1]-10), 
                    (icon_center[0]+10, icon_center[1]+10), (255, 255, 255), 2)
            cv2.line(frame, (icon_center[0]-10, icon_center[1]+10), 
                    (icon_center[0]+10, icon_center[1]-10), (255, 255, 255), 2)
    
    # Draw line thickness indicator
    cv2.putText(frame, f"Line: {line_thickness}px", 
               (ui_margin, tools_y + tool_box_size + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def check_ui_interaction(point):
    """Check if the user is interacting with UI elements"""
    global current_color_index, current_tool, drawing_color
    
    # Check color palette interaction
    for i, color in enumerate(colors):
        x = ui_margin + i * (color_box_size + ui_margin)
        y = color_palette_y
        if x <= point[0] <= x + color_box_size and y <= point[1] <= y + color_box_size:
            current_color_index = i
            drawing_color = colors[current_color_index]
            return True
    
    # Check tool selection interaction
    for i, tool in enumerate(tools):
        x = ui_margin + i * (tool_box_size + ui_margin)
        y = tools_y
        if x <= point[0] <= x + tool_box_size and y <= point[1] <= y + tool_box_size:
            current_tool = tool
            return True
    
    return False

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
        
        # Resize frame to match canvas size
        frame = cv2.resize(frame, (canvas_width, canvas_height))
        h, w, _ = frame.shape
        
        # Create UI overlay frame
        ui_overlay = np.zeros_like(frame)
        
        # Convert the BGR image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(rgb)
        
        # Convert back to BGR for display
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Create a composite view: when drawing, show black background with drawings
        # Otherwise overlay canvas on webcam frame with transparency
        if is_drawing:
            # Show black background with drawings (for better contrast while drawing)
            display_frame = drawing_background.copy()
            # Overlay canvas directly without transparency
            mask = (canvas > 0).astype(bool)
            display_frame[mask] = canvas[mask]
        else:
            # Use webcam feed with transparent overlay
            display_frame = frame.copy()
            alpha = 0.5
            mask = (canvas > 0).astype(float)
            for c in range(0, 3):
                display_frame[:, :, c] = display_frame[:, :, c] * (1 - mask[:, :, c] * alpha) + canvas[:, :, c] * mask[:, :, c] * alpha
        
        # Replace frame with our display frame
        frame = display_frame
        
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
                
                # Apply smoothing to cursor point
                smoothed_x, smoothed_y = apply_smoothing((int(cursor_point[0]), int(cursor_point[1])))
                cursor_point = (smoothed_x, smoothed_y)
                
                # Draw cursor point
                cv2.circle(frame, cursor_point, 10, (0, 0, 255), -1)
                
                # Handle gestures with cooldown to prevent rapid triggering
                if last_gesture != gesture or (current_time - gesture_start_time) > gesture_cooldown:
                    last_gesture = gesture
                    gesture_start_time = current_time
                    
                    # Check if interacting with UI
                    if not check_ui_interaction(cursor_point):
                        if gesture == "DRAW":
                            is_drawing = True
                            last_point = cursor_point
                            status_text = "Drawing"
                            status_color = (0, 255, 0)
                            
                        elif gesture == "HOVER":
                            is_drawing = False
                            last_point = None
                            status_text = "Hovering"
                            status_color = (255, 255, 0)
                            
                        elif gesture == "ERASER":
                            current_tool = "eraser"
                            status_text = "Eraser Selected"
                            status_color = (255, 0, 0)
                            
                        elif gesture == "COLOR_PICKER":
                            current_tool = "color_picker"
                            status_text = "Color Picker Selected"
                            status_color = (0, 0, 255)
                            
                        elif gesture == "CLEAR":
                            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                            status_text = "Canvas Cleared"
                            status_color = (255, 165, 0)
                            current_tool = "pen"  # Reset to pen tool after clearing
                
                # Handle drawing based on current tool
                if is_drawing:
                    if current_tool == "pen":
                        if last_point is not None:
                            cv2.line(canvas, last_point, cursor_point, drawing_color, line_thickness)
                        last_point = cursor_point
                        
                    elif current_tool == "eraser":
                        # Eraser is just a white (or canvas color) pen with larger thickness
                        erase_radius = line_thickness * 3
                        cv2.circle(canvas, cursor_point, erase_radius, (0, 0, 0), -1)
                        
                    elif current_tool == "color_picker":
                        # Pick color from canvas
                        if 0 <= cursor_point[0] < canvas_width and 0 <= cursor_point[1] < canvas_height:
                            picked_color = frame[cursor_point[1], cursor_point[0]]
                            if not np.array_equal(picked_color, [0, 0, 0]):  # Avoid picking black
                                drawing_color = (int(picked_color[0]), int(picked_color[1]), int(picked_color[2]))
                                status_text = f"Color picked: {drawing_color}"
                                current_tool = "pen"  # Switch back to pen after picking
        
        else:
            # No hands detected
            status_text = "No hands detected"
            status_color = (128, 128, 128)
            is_drawing = False
            last_point = None
        
        # Draw UI on overlay
        draw_ui(frame)
        
        # Draw status on frame
        draw_status(frame, status_text, status_color)
        
        # Display control instructions
        cv2.putText(frame, "Gesture Controls:", (10, h-160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Index finger: Hover cursor", (10, h-130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Pinch (thumb+index): Draw", (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Index+Middle: Eraser", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- Three fingers: Color Picker", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "- All fingers: Clear Canvas", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press '+'/'-' to change thickness, 'c' to change color, 'b' to toggle black screen", 
                   (10, h-180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 's' to save drawing, 'x' to exit", 
                   (10, h-200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add a toggle for black drawing mode
        cv2.putText(frame, "Drawing Mode: " + ("BLACK SCREEN" if is_drawing else "CAMERA VIEW"), 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the resulting frame
        cv2.imshow('Hand Gesture Drawing System', frame)
        
        # Key controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('+') or key == ord('='):
            line_thickness = min(line_thickness + 1, 20)
        elif key == ord('-') or key == ord('_'):
            line_thickness = max(line_thickness - 1, 1)
        elif key == ord('c'):
            # Cycle through colors
            current_color_index = (current_color_index + 1) % len(colors)
            drawing_color = colors[current_color_index]
        elif key == ord('b'):
            # Toggle black background permanently
            is_drawing = not is_drawing
        elif key == ord('s'):
            # Save the canvas
            timestamp = int(time.time())
            filename = f"drawing_{timestamp}.png"
            # Create a solid black background for the saved image
            save_image = drawing_background.copy()
            mask = (canvas > 0).astype(bool)
            save_image[mask] = canvas[mask]
            cv2.imwrite(filename, save_image)
            status_text = f"Saved as {filename}"
            status_color = (0, 255, 0)

# Clean up
video.release()
cv2.destroyAllWindows()
