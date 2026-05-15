import cv2
import multiprocessing
import time
import sys
import os
import traceback

def detect_gesture(hand_landmarks):
    """Detect thumbs up or palm gesture from hand landmarks with lenient conditions"""
    # Get finger states (up/down)
    fingers_up = [False] * 5
    
    # Thumb detection with lenient conditions
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_cmc = hand_landmarks.landmark[1]
    wrist = hand_landmarks.landmark[0]
    
    # More lenient thumb up detection
    # Only check if thumb is generally pointing up and outward
    thumb_up = (
        thumb_tip.y < thumb_mcp.y and  # Thumb tip is above MCP joint
        thumb_tip.x > thumb_cmc.x and  # Thumb is pointing outward
        abs(thumb_tip.x - thumb_ip.x) < 0.4  # More lenient straightness check
    )
    fingers_up[0] = thumb_up
    
    # Other fingers detection with lenient conditions
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
    finger_pips = [6, 10, 14, 18]  # Second joint of each finger
    finger_mcps = [5, 9, 13, 17]   # Base joint of each finger
    
    STRAIGHTNESS_TOLERANCE = 0.5  # horizontal deviation allowance
    MIN_HEIGHT_MARGIN = 0.02      # extra vertical gap to avoid noise

    finger_tip_positions = []

    for idx, (tip_id, pip_id, mcp_id) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        
        # More lenient finger up detection
        # Only check if finger is generally pointing up
        finger_up = (
            tip.y < (mcp.y - MIN_HEIGHT_MARGIN) and  # Tip is above MCP joint with margin
            abs(tip.x - pip.x) < STRAIGHTNESS_TOLERANCE  # More lenient straightness check
        )
        fingers_up[idx + 1] = finger_up
        finger_tip_positions.append((tip.x, tip.y))
    
    # Count how many fingers are up
    fingers_up_count = sum(fingers_up[1:])
    
    # Debug print for finger states
    print(f"Finger states - Thumb: {fingers_up[0]}, Others: {fingers_up[1:]} (Count: {fingers_up_count})")
    
    # Check for thumbs up first - more lenient conditions
    if (fingers_up[0] and  # Thumb is up
        fingers_up_count <= 1 and  # At most 1 other finger up
        thumb_tip.y < wrist.y):  # Thumb is above wrist
        print("Thumbs up detected!")
        return "send"
    
    # More lenient palm detection
    # Consider it palm if:
    # 1. At least 3 fingers are up (excluding thumb) OR
    # 2. At least 2 fingers are up and the spread between them is wide (open palm)
    # 3. Thumb state is ignored unless clearly also signaling thumbs-up
    elif fingers_up_count >= 3 or (
        fingers_up_count >= 2 and
        finger_tip_positions and
        (max(x for x, _ in finger_tip_positions) - min(x for x, _ in finger_tip_positions)) > 0.18
    ):
        print("Palm detected!")
        return "cancel"
    
    return None

def init_mediapipe():
    """Initialize MediaPipe components"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Installed mediapipe package does not expose mp.solutions. "
                "Install a compatible version: python -m pip install \"mediapipe==0.10.13\""
            )

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.1
        )
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        return mp_hands, hands, mp_draw, mp_drawing_styles
    except Exception as e:
        print(f"❌ Error initializing MediaPipe: {e}")
        raise

def list_available_cameras():
    """List all available cameras and their properties"""
    available_cameras = []
    for device_id in range(10):  # Check more camera indices
        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    camera_info = {
                        'device_id': device_id,
                        'resolution': f"{int(width)}x{int(height)}",
                        'fps': fps,
                        'frame_shape': frame.shape
                    }
                    available_cameras.append(camera_info)
                    print(f"✅ Found camera {device_id}: {camera_info}")
                cap.release()
        except Exception as e:
            print(f"Failed to check camera device {device_id}: {e}")
    return available_cameras

def gesture_recognition_process(gesture_queue, stop_event):
    """Process for running gesture recognition"""
    cap = None
    try:
        print("\n🎥 Starting gesture recognition process...")
        
        # Import MediaPipe here in the child process
        import mediapipe as mp
        
        # Initialize MediaPipe first
        try:
            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    "module 'mediapipe' has no attribute 'solutions'. "
                    "Please install a compatible version: python -m pip install \"mediapipe==0.10.13\""
                )

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.1
            )
            mp_draw = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            print("✅ MediaPipe Hands initialized")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            return
        
        # List and select camera
        print("\nScanning for available cameras...")
        available_cameras = list_available_cameras()
        
        if not available_cameras:
            print("❌ No cameras found")
            print("Please ensure camera permissions are granted in System Preferences > Security & Privacy > Privacy > Camera")
            return
        
        # Try to use the first available camera
        selected_camera = available_cameras[0]
        print(f"\nSelected camera: {selected_camera}")
        
        try:
            cap = cv2.VideoCapture(selected_camera['device_id'])
            if not cap.isOpened():
                print(f"❌ Failed to open camera device {selected_camera['device_id']}")
                return
            
            # Try to read a test frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("❌ Failed to read test frame")
                return
                
            print(f"✅ Camera opened successfully:")
            print(f"  - Device ID: {selected_camera['device_id']}")
            print(f"  - Resolution: {selected_camera['resolution']}")
            print(f"  - FPS: {selected_camera['fps']}")
            print(f"  - Frame shape: {test_frame.shape}")
            
            # Set camera properties for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return
        
        # Wait for camera to initialize
        time.sleep(2)
        
        # Create window with explicit flags
        window_name = 'Gesture Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.moveWindow(window_name, 100, 100)
        
        print("\n👋 Ready for gestures!")
        print("Available gestures:")
        print("1. 👍 Thumbs up  -> send email")
        print("2. ✋ Palm       -> cancel email")
        
        last_gesture = None
        gesture_counter = 0
        GESTURE_THRESHOLD = 1  # Reduced threshold for faster detection
        frame_count = 0
        last_frame_time = time.time()
        gesture_detected = False
        gesture_start_time = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("❌ Camera read error - trying to reconnect...")
                time.sleep(1)
                continue
            
            # Calculate and print FPS
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print status every 30 frames
                print(f"📸 Processing frame {frame_count} at {fps:.1f} FPS")
            
            # Resize frame if needed
            if frame.shape[1] > 640:  # If width is larger than 640
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks with improved visibility
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks with custom style
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Detect gesture
                    current_gesture = detect_gesture(hand_landmarks)
                    
                    # Gesture smoothing with improved stability
                    if current_gesture == last_gesture:
                        gesture_counter += 1
                        if gesture_counter == 1:
                            print("🔍 Potential gesture detected, hold steady...")
                            gesture_start_time = time.time()
                    else:
                        gesture_counter = max(0, gesture_counter - 1)  # Gradual decrease
                        gesture_detected = False
                    
                    if gesture_counter >= GESTURE_THRESHOLD and current_gesture:
                        if not gesture_detected:
                            print(f"✨ Gesture detected: {current_gesture}")
                            gesture_queue.put(current_gesture)
                            gesture_detected = True
                            # Add a small delay after gesture detection
                            time.sleep(0.2)  # Reduced delay for faster response
                    
                    last_gesture = current_gesture
                    
                    # Draw gesture progress bar with improved visibility
                    if gesture_counter > 0:
                        progress = min(1.0, gesture_counter / GESTURE_THRESHOLD)
                        bar_width = int(frame.shape[1] * progress)
                        # Draw background
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)
                        # Draw progress
                        cv2.rectangle(frame, (0, 0), (bar_width, 20), (0, 255, 0), -1)
                        # Add percentage text
                        cv2.putText(frame, f"{int(progress * 100)}%", (10, 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show gesture name when detected
                    if gesture_detected:
                        gesture_text = "THUMBS UP - Sending" if current_gesture == "send" else "PALM - Cancelling"
                        # Draw background for text
                        cv2.rectangle(frame, (0, 30), (400, 60), (0, 0, 0), -1)
                        cv2.putText(frame, gesture_text, (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                gesture_counter = max(0, gesture_counter - 1)  # Gradual decrease
                gesture_detected = False
            
            # Add gesture guide text with improved visibility
            cv2.rectangle(frame, (0, frame.shape[0] - 60), (400, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, "Show gesture and hold steady", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Thumbs up = Send  |  Palm = Cancel", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame counter and FPS
            cv2.putText(frame, f"Frame: {frame_count} | FPS: {fps:.1f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, cv2.flip(frame, 1))
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("👋 Gesture recognition stopped by user")
                break
    
    except Exception as e:
        print(f"❌ Gesture recognition error: {e}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
    
    finally:
        print("\n🎥 Cleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # Extra cleanup for macOS
        for i in range(4):
            cv2.waitKey(1)
        if 'hands' in locals():
            hands.close()

if __name__ == "__main__":
    # Test the gesture recognition
    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    
    try:
        gesture_recognition_process(queue, stop_event)
    except KeyboardInterrupt:
        print("\nStopping gesture recognition...") 