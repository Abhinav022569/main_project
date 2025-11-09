import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np 

mp_solutions = mp.solutions

# --- Constants for our logic ---

# --- Camera Setup ---
PHONE_CAM_INDEX = 1  # The index of your DroidCam (usually 1 or 2)
LAPTOP_CAM_INDEX = 0 # The index of your built-in webcam (usually 0)

# --- Gesture & Movement Tuning ---
SMOOTHING_FACTOR = 0.8    # 0.0=no smooth, 1.0=no movement.
CLICK_THRESHOLD = 0.08    # Normalized distance for a pinch
ACTIVATION_TIMEOUT = 1.5  # Seconds to wait for the next gesture in the sequence
SENSITIVITY = 2.6         # Multiplies mouse movement. Higher = faster.
RESIZE_SENSITIVITY_THRESHOLD = 0.03 # How much hand distance must change to trigger a resize key press

# --- Camera Health Tuning ---
VIDEO_HEALTH_THRESHOLD = 15 # How much "detail" a camera feed needs (avoids pure black)
VIDEO_MOVEMENT_THRESHOLD = 5 # How much "movement" is needed between frames (avoids static images)

# --- Prevent pyautogui from crashing if the mouse goes to a corner ---
pyautogui.FAILSAFE = False

# --- Helper function for distance ---
def get_distance(p1, p2):
    """Calculates the 2D Euclidean distance between two MediaPipe landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- Helper functions for gesture recognition ---
def is_thumbs_up(hand_landmarks):
    """Checks if the hand is in a 'Thumbs Up' gesture."""
    thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_IP] # Inner thumb joint
    
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP] # Middle index joint
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    # Rule: Thumb is extended (tip is above inner joint)
    # AND all other fingers are closed (tips are below middle joints)
    thumb_extended = thumb_tip.y < thumb_ip.y
    fingers_closed = (index_tip.y > index_pip.y) and (middle_tip.y > middle_pip.y)
    
    return thumb_extended and fingers_closed

def is_palm_splayed(hand_landmarks):
    """Checks if the hand is in a 'Splayed Palm' gesture."""
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_PIP]
    
    # Rule: All 4 fingers are extended (tips are above middle joints)
    fingers_extended = (index_tip.y < index_pip.y) and \
                        (middle_tip.y < middle_pip.y) and \
                        (ring_tip.y < ring_pip.y) and \
                        (pinky_tip.y < pinky_pip.y)
    
    return fingers_extended

def is_fist(hand_landmarks):
    """Checks if the hand is in a 'Fist' gesture."""
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_PIP]
    
    # Rule: All 4 fingers are closed (tips are below middle joints)
    fingers_closed = (index_tip.y > index_pip.y) and \
                        (middle_tip.y > middle_pip.y) and \
                        (ring_tip.y > ring_pip.y) and \
                        (pinky_tip.y > pinky_pip.y)
                        
    return fingers_closed

# --- NEW: is_pointing HELPER FUNCTION ---
def is_pointing(hand_landmarks):
    """Checks if the hand is in a 'Pointing' gesture (index extended, others closed)."""
    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_PIP]
    
    # Rule: Index is extended (tip is above middle joint)
    # AND all other fingers are closed (tips are below middle joints)
    index_extended = index_tip.y < index_pip.y
    other_fingers_closed = (middle_tip.y > middle_pip.y) and \
                           (ring_tip.y > ring_pip.y) and \
                           (pinky_tip.y > pinky_pip.y)
                           
    return index_extended and other_fingers_closed
# --- END NEW HELPER ---

# --- NEW: ROBUST Helper function for camera health ---
def check_camera_health(cam):
    """
    Tries to read two frames and checks if it's a real, MOVING image (not a blank/static screen).
    """
    # 1. Read first frame
    success1, frame1 = cam.read()
    if not success1 or frame1 is None:
        return False, None # Failed to read

    # 2. Check if it's a blank screen (like pure black)
    if np.std(frame1) < VIDEO_HEALTH_THRESHOLD:
        return False, frame1 # It's just a blank color
        
    # 3. Read second frame after a short delay
    time.sleep(0.05) # 50ms delay
    success2, frame2 = cam.read()
    if not success2 or frame2 is None:
        return False, None
        
    # 4. Check if the image is static (like the orange screen)
    frame_diff = cv2.absdiff(frame1, frame2)
    if np.std(frame_diff) < VIDEO_MOVEMENT_THRESHOLD:
        return False, frame2 # It's a static image
        
    return True, frame2 # Success! It's a real, moving video feed.
# --- END NEW HELPER ---

# --- Setup for webcam and models ---
mp_hands = mp_solutions.hands
# --- UPDATED: Now looks for 2 hands ---
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp_solutions.drawing_utils
# --- END OF SETUP ---


# --- AUTOMATIC FALLBACK CAMERA LOGIC ---
print(f"Attempting to connect to phone camera (Index {PHONE_CAM_INDEX})...")
cam = cv2.VideoCapture(PHONE_CAM_INDEX)

is_healthy, test_frame = check_camera_health(cam)

if not is_healthy:
    print(f"Phone camera failed or sent a blank/static screen.")
    print(f"FALLING BACK to laptop camera (Index {LAPTOP_CAM_INDEX})...")
    cam.release() # Release the failed camera
    cam = cv2.VideoCapture(LAPTOP_CAM_INDEX)
    
    # Use a SIMPLE check for the laptop camera (don't check for static frames)
    is_healthy, test_frame = cam.read()
    
    if not is_healthy:
        print("--- FATAL ERROR ---")
        print("Failed to connect to *any* camera.")
        print("Please check your webcam drivers.")
        print("-------------------")
        cam.release()
        exit()
    else:
        print(f"Successfully connected to laptop camera.")
else:
    print(f"Successfully connected to phone camera.")
# --- END OF LOGIC ---

screen_width, screen_height = pyautogui.size()

# --- 1. Right Hand: Smoothing & Relative Mode Variables ---
smooth_x, smooth_y = 0, 0
prev_palm_x, prev_palm_y = 0, 0
first_active_frame = True # Flag to "prime" the relative movement

# --- 2. State Machine Variables (Now for BOTH hands) ---
is_right_active = False           
is_left_active = False
current_state_right = "IDLE"
current_state_left = "IDLE"
last_gesture_time_right = 0
last_gesture_time_left = 0

# --- 3. Click Lock Variable ---
click_lock = False

# --- 4. Resize Mode Variables ---
resize_mode_active = False
initial_fist_distance = 0

print("Air Mouse v3 (Dual Hand) Running. Show 'Thumbs Up' then 'Palm' (per hand) to activate.")
print("Press 'q' to quit.")

while True:
    
    # We already read one frame during the health check, so we can use it
    if test_frame is not None:
        frame = test_frame
        test_frame = None # Clear it so we only use it once
    else:
        success, frame = cam.read()
        if not success:
            continue
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # --- NEW: Hand Separation Logic ---
    left_hand_landmarks = None
    right_hand_landmarks = None
    left_hand_visible = False
    right_hand_visible = False
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            if handedness == "Right":
                right_hand_landmarks = hand_landmarks
                right_hand_visible = True
            elif handedness == "Left":
                left_hand_landmarks = hand_landmarks
                left_hand_visible = True
        
        # Draw all visible hands
        for hand_landmarks in results.multi_hand_landmarks:
             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # --- NEW: Dual State Machines ---
    
    # --- Right Hand FSM ---
    if right_hand_visible:
        thumb_is_up_right = is_thumbs_up(right_hand_landmarks)
        palm_is_splayed_right = is_palm_splayed(right_hand_landmarks)

        if current_state_right == "IDLE":
            if thumb_is_up_right:
                print("RIGHT HAND: Saw Thumbs Up. Awaiting Palm...")
                current_state_right = "AWAITING_PALM"
                last_gesture_time_right = time.time()
                
        elif current_state_right == "AWAITING_PALM":
            if time.time() - last_gesture_time_right > ACTIVATION_TIMEOUT:
                current_state_right = "IDLE"
            elif palm_is_splayed_right:
                is_right_active = not is_right_active
                print(f"RIGHT HAND: Saw Palm! Mouse is now {'ACTIVE' if is_right_active else 'INACTIVE'}")
                if is_right_active:
                    first_active_frame = True # Prime the relative movement
                current_state_right = "IDLE"
    
    # --- Left Hand FSM ---
    if left_hand_visible:
        thumb_is_up_left = is_thumbs_up(left_hand_landmarks)
        palm_is_splayed_left = is_palm_splayed(left_hand_landmarks)

        if current_state_left == "IDLE":
            if thumb_is_up_left:
                print("LEFT HAND: Saw Thumbs Up. Awaiting Palm...")
                current_state_left = "AWAITING_PALM"
                last_gesture_time_left = time.time()
                
        elif current_state_left == "AWAITING_PALM":
            if time.time() - last_gesture_time_left > ACTIVATION_TIMEOUT:
                current_state_left = "IDLE"
            elif palm_is_splayed_left:
                is_left_active = not is_left_active
                print(f"LEFT HAND: Saw Palm! Hand is now {'ACTIVE' if is_left_active else 'INACTIVE'}")
                current_state_left = "IDLE"

    # --- End of State Machines ---

    # --- Feature: Resize Mode (NEW) ---
    # This logic overrides all other controls if active
    
    if is_right_active and is_left_active and right_hand_visible and left_hand_visible:
        
        is_right_fist = is_fist(right_hand_landmarks)
        is_left_fist = is_fist(left_hand_landmarks)
        
        palm_center_right = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        palm_center_left = left_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        if is_right_fist and is_left_fist:
            current_fist_distance = get_distance(palm_center_right, palm_center_left)
            
            if not resize_mode_active:
                # --- ENTER RESIZE MODE ---
                print("RESIZE MODE: Activated! Pull hands apart/together.")
                resize_mode_active = True
                initial_fist_distance = current_fist_distance
                
                # Send OS commands to enter "Size" mode
                pyautogui.keyDown('alt')
                pyautogui.press('space')
                pyautogui.keyUp('alt')
                pyautogui.press('s')
                
            else:
                # --- RESIZE MODE IS RUNNING ---
                distance_delta = current_fist_distance - initial_fist_distance
                
                # Check for "move apart"
                if distance_delta > RESIZE_SENSITIVITY_THRESHOLD:
                    print("RESIZE: WIDER")
                    pyautogui.press('right')
                    initial_fist_distance = current_fist_distance # Reset delta
                
                # Check for "move together"
                elif distance_delta < -RESIZE_SENSITIVITY_THRESHOLD:
                    print("RESIZE: NARROWER")
                    pyautogui.press('left')
                    initial_fist_distance = current_fist_distance # Reset delta

        elif resize_mode_active:
            # --- EXIT RESIZE MODE (fists are undone) ---
            print("RESIZE MODE: Deactivated.")
            resize_mode_active = False
            pyautogui.press('enter') # Send "Enter" to confirm the new size
            
    # --- Mouse Control (Only if RIGHT hand is active AND resize mode is OFF) ---
    if is_right_active and right_hand_visible and not resize_mode_active:
        
        # --- NEW: Check for "Pointing" gesture ---
        if is_pointing(right_hand_landmarks):
            
            # --- Feature 1: Relative Movement Logic ---
            # We'll use the index finger tip for movement now, as it's the pointer
            pointer_tip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP] # Landmark #8
            index_tip = pointer_tip # For pinch-to-click
            
            # On the first frame, "prime" the system
            if first_active_frame:
                prev_palm_x, prev_palm_y = pointer_tip.x, pointer_tip.y
                smooth_x, smooth_y = pyautogui.position() 
                first_active_frame = False
            else:
                delta_x = (pointer_tip.x - prev_palm_x) * screen_width * SENSITIVITY
                delta_y = (pointer_tip.y - prev_palm_y) * screen_height * SENSITIVITY
                
                target_x = smooth_x + delta_x
                target_y = smooth_y + delta_y
                
                smooth_x = (smooth_x * SMOOTHING_FACTOR) + (target_x * (1 - SMOOTHING_FACTOR))
                smooth_y = (smooth_y * SMOOTHING_FACTOR) + (target_y * (1 - SMOOTHING_FACTOR))
                
                pyautogui.moveTo(smooth_x, smooth_y)
                
                prev_palm_x, prev_palm_y = pointer_tip.x, pointer_tip.y
            # --- End of Relative Movement ---

            # --- Feature 3: Pinch-to-Click (still uses index finger) ---
            thumb_tip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
            wrist = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            middle_pip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
            hand_size = get_distance(wrist, middle_pip)
            
            if hand_size > 0.01: # Avoid division by zero
                pinch_distance = get_distance(index_tip, thumb_tip) / hand_size
            
                if pinch_distance < CLICK_THRESHOLD:
                    if not click_lock:
                        print("CLICK!")
                        pyautogui.click()
                        click_lock = True
                
                if pinch_distance > (CLICK_THRESHOLD * 1.5):
                    click_lock = False

        else:
            # --- NEW: Hand is active, but not pointing ---
            # Reset the 'first_active_frame' flag.
            # This ensures that when we start pointing again, it re-primes from the mouse's current position.
            first_active_frame = True
            
            # Check for pinch-to-click even when not moving
            index_tip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP] # Landmark #8
            thumb_tip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
            wrist = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            middle_pip = right_hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
            hand_size = get_distance(wrist, middle_pip)

            if hand_size > 0.01: 
                pinch_distance = get_distance(index_tip, thumb_tip) / hand_size
            
                if pinch_distance < CLICK_THRESHOLD:
                    if not click_lock:
                        print("CLICK! (while stationary)")
                        pyautogui.click()
                        click_lock = True
                
                if pinch_distance > (CLICK_THRESHOLD * 1.5):
                    click_lock = False
        
    # --- Drawing and Display (Always on) ---
    
    # Draw status text
    status_right = f"Right: {'ACTIVE' if is_right_active else 'INACTIVE'}"
    color_right = (0, 255, 0) if is_right_active else (0, 0, 255)
    cv2.putText(frame, status_right, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)

    status_left = f"Left: {'ACTIVE' if is_left_active else 'INACTIVE'}"
    color_left = (0, 255, 0) if is_left_active else (0, 0, 255)
    cv2.putText(frame, status_left, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)
    
    if resize_mode_active:
        cv2.putText(frame, "RESIZE MODE ON", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    cv2.imshow("Air Mouse v3 - Dual Hand", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()