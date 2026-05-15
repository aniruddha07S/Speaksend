from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import threading
import time
import os
from pathlib import Path
import logging
import multiprocessing as mp
import cv2
import numpy as np
from collections import deque
from gesture_process import detect_gesture
from gmail_service import send_email_via_gmail
from semantic_search import run_semantic_search, get_most_recent_email

# Import the VoiceEmailSystem from app.py
from app import VoiceEmailSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
is_listening = False
current_state = "welcome"
current_username = None
voice_system = None
listening_thread = None
video_lock = threading.Lock()
camera_enabled = True
manual_command_queue = deque()
typed_input_queue = deque()
manual_command_lock = threading.Lock()
typed_input_lock = threading.Lock()
listener_lock = threading.Lock()
connected_clients = 0
mic_pause_until = 0.0

class WebVoiceEmailSystem(VoiceEmailSystem):
    """VoiceEmailSystem wrapper that always emits prompts to SocketIO."""
    
    def __init__(self):
        super().__init__()
        # Backend TTS inside worker threads can hang on Windows (pyttsx3/comtypes).
        # In web mode, prefer browser speech synthesis and keep backend TTS optional.
        self.enable_backend_tts = os.getenv("VOICE_EMAIL_WEB_BACKEND_TTS", "0") == "1"
        self.recent_email_offset = 0

    def reset_recent_cursor(self):
        self.recent_email_offset = 0

    def speak(self, text):
        """Emit prompt to frontend first, then do platform-specific TTS."""
        socketio.emit('speak', {'text': text})
        if self.enable_backend_tts:
            super().speak(text)

    def recognize_speech(self, prompt=None, timeout=None, phrase_time_limit=None):
        """Recognize speech and mirror heard text to frontend chat."""
        global mic_pause_until
        # Typed chat input should behave like spoken replies for any active prompt.
        with typed_input_lock:
            if typed_input_queue:
                text = typed_input_queue.popleft()
                socketio.emit('heard_text', {'text': text})
                return text

        # Allow frontend/gesture commands to drive the same backend flow.
        with manual_command_lock:
            if manual_command_queue:
                text = manual_command_queue.popleft()
                socketio.emit('heard_text', {'text': text})
                return text

        # If user typed/clicked a command, briefly pause mic capture to avoid races.
        if time.time() < mic_pause_until:
            return None

        text = super().recognize_speech(prompt=prompt, timeout=timeout, phrase_time_limit=phrase_time_limit)
        if text:
            socketio.emit('heard_text', {'text': text})
        return text

    def wait_for_typed_input(self, prompt, timeout=120):
        """Wait for typed chat input only (no microphone), with cancel support."""
        self.speak(prompt)
        start = time.time()
        while time.time() - start < timeout:
            with typed_input_lock:
                if typed_input_queue:
                    text = str(typed_input_queue.popleft()).strip()
                    if text:
                        socketio.emit('heard_text', {'text': text})
                        if text.lower() in ("cancel", "stop", "exit"):
                            return None
                        return text

            # Allow manual command queue to cancel typed-only prompts.
            with manual_command_lock:
                if manual_command_queue:
                    command = str(manual_command_queue.popleft()).strip().lower()
                    socketio.emit('heard_text', {'text': command})
                    if command in ("cancel", "stop", "exit"):
                        return None

            time.sleep(0.1)
        return None

    def get_email_details(self):
        """Web-mode compose details: body is typed in chat (no mic required)."""
        # Recipient (voice or typed text both work via recognize_speech).
        for _ in range(self.max_retries):
            self.speak("Please say or type the recipient's email address.")
            email = self.recognize_speech(timeout=15, phrase_time_limit=18)
            if not email:
                self.speak("I didn't hear an email address. Let's try again.")
                continue

            email = email.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
            self.speak(f"I heard the email address as: {email}")
            self.speak("To confirm this email address, say or type 'confirm'. To try again, say or type 'cancel'.")
            confirmation = self.recognize_speech(timeout=8, phrase_time_limit=4)

            if confirmation and "confirm" in confirmation:
                break
            if confirmation and "cancel" in confirmation and _ < self.max_retries - 1:
                self.speak("Let's try again.")
                continue
        else:
            self.speak("Too many unsuccessful attempts. Cancelling email composition.")
            return None, None, None

        # Subject (voice or typed text both work).
        self.speak("Please dictate or type the subject of your email.")
        subject = self.recognize_speech(timeout=20, phrase_time_limit=20)
        if not subject:
            return None, None, None

        # Body (typed chat only, as requested).
        body = self.wait_for_typed_input(
            "Please type your email content in the chat box and press Send. "
            "You can type cancel to stop."
        )
        if not body:
            return None, None, None

        if body.lower().startswith("body:"):
            body = body[5:].strip()

        return email, subject, body

    def handle_email_composition(self, username):
        """Web-mode compose flow that accepts queued/manual/gesture confirm/cancel."""
        self.speak("Let's compose your email.")
        email_details = self.get_email_details()
        if (
            not email_details
            or len(email_details) != 3
            or any(part is None or str(part).strip() == "" for part in email_details)
        ):
            self.speak("Email composition cancelled.")
            socketio.emit("compose_card", {"status": "cancelled"})
            return False

        email, subject, body = email_details
        socketio.emit("compose_card", {
            "status": "draft",
            "to": email,
            "subject": subject,
            "body": body
        })
        self.speak("Here's your email. Please review:")
        self.speak(f"To: {email}")
        self.speak(f"Subject: {subject}")
        self.speak(f"Content: {body}")
        self.speak("You can confirm by voice or gesture. Say confirm to send, or cancel to stop.")

        start_time = time.time()
        timeout = 30
        while time.time() - start_time < timeout:
            response = self.recognize_speech(timeout=2, phrase_time_limit=3)
            if not response:
                continue

            if "edit" in response:
                updated_body = self.wait_for_typed_input(
                    "Please type the updated email content in chat and press Send. "
                    "Type cancel to keep current draft."
                )
                if updated_body:
                    body = updated_body
                    socketio.emit("compose_card", {
                        "status": "draft",
                        "to": email,
                        "subject": subject,
                        "body": body
                    })
                    self.speak("Email content updated. Say confirm to send, or cancel to stop.")
                else:
                    self.speak("Keeping your current draft. Say confirm to send, or cancel to stop.")
                continue

            if "confirm" in response:
                self.speak("Confirmation received. Sending email now...")
                try:
                    send_email_via_gmail(email, subject, body)
                    self.speak("Email sent successfully!")
                    socketio.emit("compose_card", {
                        "status": "sent",
                        "to": email,
                        "subject": subject,
                        "body": body
                    })
                    return True
                except Exception as e:
                    print(f"\n❌ Error sending email: {e}")
                    self.speak("Sorry, I encountered an error while sending your email.")
                    socketio.emit("compose_card", {
                        "status": "error",
                        "to": email,
                        "subject": subject,
                        "body": body,
                        "error": str(e)
                    })
                    return False

            if "cancel" in response:
                self.speak("Cancelling email as requested.")
                socketio.emit("compose_card", {
                    "status": "cancelled",
                    "to": email,
                    "subject": subject,
                    "body": body
                })
                return False

        self.speak("No confirmation received within time limit. Cancelling email.")
        socketio.emit("compose_card", {
            "status": "timeout",
            "to": email,
            "subject": subject,
            "body": body
        })
        return False

    def handle_email_search(self):
        """Web-mode search with card output for frontend."""
        self.speak("What kind of emails would you like to search for?")
        query = self.recognize_speech()
        if not query:
            self.speak("No search query provided.")
            return False

        try:
            matches = run_semantic_search(query)
            cards = []
            if matches:
                for item in matches[:5]:
                    email = item["email"]
                    cards.append({
                        "from": email.get("from", ""),
                        "subject": email.get("subject", ""),
                        "body": email.get("body", ""),
                        "date": email.get("date", ""),
                        "similarity": float(item.get("similarity", 0.0))
                    })
                self.speak(f"I found {len(cards)} matching emails.")
            else:
                self.speak("No matching emails found.")
            socketio.emit("search_cards", {"query": query, "emails": cards})
            return True
        except Exception as e:
            self.speak("Sorry, I encountered an error while searching emails.")
            socketio.emit("search_cards", {"query": query, "emails": [], "error": str(e)})
            return False

    def handle_most_recent_email(self):
        """Web-mode recent email sequence (newest -> older)."""
        self.speak("Fetching your recent email...")
        try:
            result = get_most_recent_email(offset=self.recent_email_offset)
            if result:
                _, email_content = result
                card = {
                    "from": email_content.get("from", ""),
                    "subject": email_content.get("subject", ""),
                    "body": email_content.get("body", ""),
                    "date": email_content.get("date", "")
                }
                self.recent_email_offset += 1
                socketio.emit("recent_card", {"email": card})
                self.speak("Here is your recent email.")
                return True
            if self.recent_email_offset > 0:
                self.speak("No more recent emails found. Starting from the newest again.")
                self.recent_email_offset = 0
            else:
                self.speak("Sorry, I couldn't fetch your recent email.")
            socketio.emit("recent_card", {"email": None})
            return False
        except Exception as e:
            self.speak("Sorry, I encountered an error while fetching your recent email.")
            socketio.emit("recent_card", {"email": None, "error": str(e)})
            return False
    
    def handle_voice_enrollment(self, username):
        """Override to set speak function on authenticator - matches app.py exactly"""
        self.speak(f"Hello {username}. Let's create your voice profile.")
        
        from voice_auth import VoiceAuthenticator
        authenticator = VoiceAuthenticator()
        authenticator.set_speak_function(self.speak)
        
        if authenticator.enroll_user(username):
            return True
        else:
            self.speak("There was a problem creating your voice profile.")
            self.speak("Would you like to try again? Say yes or no.")
            retry = self.recognize_speech(timeout=5)
            if retry and 'yes' in retry.lower():
                return self.handle_voice_enrollment(username)
            return False
    
    def authenticate_user(self, username):
        """Override to set speak function on authenticator - matches app.py exactly"""
        from voice_auth import VoiceAuthenticator
        authenticator = VoiceAuthenticator()
        authenticator.set_speak_function(self.speak)
        
        # Check if profile exists
        profile_path = authenticator.voice_dir / f"{username}_profile.pkl"
        if not profile_path.exists():
            self.speak(f"No voice profile found for username: {username}")
            self.speak("Would you like to create a new profile? Say 'create' or 'skip'.")
            response = self.recognize_speech(timeout=5)
            if response and 'create' in response:
                return self.handle_voice_enrollment(username)
            return False
            
        # Verify voice
        for attempt in range(3):
            self.speak(f"Please verify your voice for username: {username}")
            if authenticator.verify_user(username):
                self.speak("Voice authentication successful!")
                return True
            else:
                if attempt < 2:
                    self.speak("Voice authentication failed. Say 'continue' to try again or 'stop' to cancel.")
                    response = self.recognize_speech(timeout=5)
                    if not response or 'stop' in response:
                        return False
                else:
                    self.speak("Voice authentication failed too many times.")
        return False

def handle_username_flow():
    """Handle username capture and authentication flow - matches app.py exactly"""
    global current_state, current_username, voice_system
    
    # Retry logic to avoid infinite loops
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        # Speak welcome message only on first attempt
        if attempt == 0:
            voice_system.speak("Welcome to Voice Email System!")
            time.sleep(0.5)  # Brief pause after welcome
        
        # Get username (this already speaks the prompt)
        username = voice_system.get_username()
        if username:
            break  # Successfully got username
        else:
            attempt += 1
            if attempt < max_attempts:
                voice_system.speak("Let's try again.")
                time.sleep(0.5)
            else:
                voice_system.speak("I couldn't catch a username. Please refresh and try again.")
                current_state = "welcome"
                socketio.emit('state_change', {'state': 'welcome'})
                current_username = None
                return
    
    if not username:
        return
    
    current_username = username
    
    # Check if user exists
    profile_path = Path("voice_auth") / f"{username}_profile.pkl"
    
    if not profile_path.exists():
        # New user - enroll voice
        current_state = "enroll"
        socketio.emit('state_change', {'state': 'enroll'})
        if not voice_system.handle_voice_enrollment(username):
            current_state = "welcome"
            socketio.emit('state_change', {'state': 'welcome'})
            current_username = None
            return
    else:
        # Existing user - verify voice
        current_state = "auth"
        socketio.emit('state_change', {'state': 'auth'})
        voice_system.speak(f"Welcome back {username}! Let's verify your voice.")
        if not voice_system.authenticate_user(username):
            current_state = "welcome"
            socketio.emit('state_change', {'state': 'welcome'})
            current_username = None
            return
    
    # Success - go to main menu
    current_state = "main_menu"
    socketio.emit('state_change', {'state': 'main_menu'})

def handle_main_menu():
    """Handle main menu commands - matches app.py exactly"""
    global current_state, current_username, voice_system, is_listening, camera_enabled
    
    # Main menu loop (matches app.py run() method)
    while current_state == "main_menu":
        voice_system.speak("What would you like to do? Say compose, search, recent, or exit.")
        command = voice_system.recognize_speech(timeout=5)
        
        if not command:
            voice_system.speak("I didn't hear a command. Please try again.")
            continue
        
        socketio.emit('voice_command', {'command': command})
        
        if command == "compose":
            current_state = "compose"
            socketio.emit('state_change', {'state': 'compose'})

            # Run compose once when entering compose mode. Only repeat if user says "compose" again.
            voice_system.handle_email_composition(current_username)
            while current_state == "compose":
                voice_system.speak("You are in compose mode. Say compose again, search, recent, cancel, or exit.")
                next_cmd = voice_system.recognize_speech(timeout=6, phrase_time_limit=6)
                if not next_cmd:
                    voice_system.speak("I didn't catch that. Please say compose, search, recent, cancel, or exit.")
                    continue
                socketio.emit('voice_command', {'command': next_cmd})
                if next_cmd == "compose":
                    voice_system.handle_email_composition(current_username)
                    continue
                if next_cmd == "search":
                    current_state = "search"
                    socketio.emit('state_change', {'state': 'search'})
                    break
                if next_cmd == "recent":
                    current_state = "recent"
                    socketio.emit('state_change', {'state': 'recent'})
                    break
                if next_cmd in ("cancel", "back", "main menu", "main_menu", "menu"):
                    current_state = "main_menu"
                    socketio.emit('state_change', {'state': 'main_menu'})
                    voice_system.speak("Returning to main menu.")
                    break
                if next_cmd == "exit":
                    voice_system.speak("Goodbye!")
                    current_state = "stopped"
                    current_username = None
                    is_listening = False
                    camera_enabled = False
                    socketio.emit('camera_state', {'enabled': camera_enabled})
                    socketio.emit('state_change', {'state': 'stopped'})
                    return
                voice_system.speak("Please say compose, search, recent, cancel, or exit.")
        elif command == "search":
            current_state = "search"
            socketio.emit('state_change', {'state': 'search'})
            while current_state == "search":
                voice_system.handle_email_search()
                voice_system.speak("You are in search mode. Say search again, compose, recent, or exit.")
                next_cmd = voice_system.recognize_speech(timeout=6, phrase_time_limit=6)
                if not next_cmd:
                    voice_system.speak("I didn't catch that. Please say search, compose, recent, or exit.")
                    continue
                socketio.emit('voice_command', {'command': next_cmd})
                if next_cmd == "search":
                    continue
                if next_cmd == "compose":
                    current_state = "compose"
                    socketio.emit('state_change', {'state': 'compose'})
                    break
                if next_cmd == "recent":
                    current_state = "recent"
                    socketio.emit('state_change', {'state': 'recent'})
                    break
                if next_cmd == "exit":
                    voice_system.speak("Goodbye!")
                    current_state = "stopped"
                    current_username = None
                    is_listening = False
                    camera_enabled = False
                    socketio.emit('camera_state', {'enabled': camera_enabled})
                    socketio.emit('state_change', {'state': 'stopped'})
                    return
                voice_system.speak("Please say search, compose, recent, or exit.")
        elif command == "recent":
            current_state = "recent"
            socketio.emit('state_change', {'state': 'recent'})
            voice_system.reset_recent_cursor()
            while current_state == "recent":
                voice_system.handle_most_recent_email()
                voice_system.speak(
                    "You are in recent mode. Say recent to hear another email, "
                    "compose, search, or exit."
                )
                next_cmd = voice_system.recognize_speech(timeout=6, phrase_time_limit=6)
                if not next_cmd:
                    voice_system.speak("I didn't catch that. Please say recent, compose, search, or exit.")
                    continue

                socketio.emit('voice_command', {'command': next_cmd})
                if next_cmd == "recent":
                    continue
                if next_cmd == "compose":
                    current_state = "compose"
                    socketio.emit('state_change', {'state': 'compose'})
                    voice_system.handle_email_composition(current_username)
                    current_state = "main_menu"
                    socketio.emit('state_change', {'state': 'main_menu'})
                    break
                if next_cmd == "search":
                    current_state = "search"
                    socketio.emit('state_change', {'state': 'search'})
                    voice_system.handle_email_search()
                    current_state = "main_menu"
                    socketio.emit('state_change', {'state': 'main_menu'})
                    break
                if next_cmd == "exit":
                    voice_system.speak("Goodbye!")
                    current_state = "stopped"
                    current_username = None
                    is_listening = False
                    camera_enabled = False
                    socketio.emit('camera_state', {'enabled': camera_enabled})
                    socketio.emit('state_change', {'state': 'stopped'})
                    return

                voice_system.speak("Please say recent, compose, search, or exit.")
        elif command == "exit":
            voice_system.speak("Goodbye!")
            current_state = "stopped"
            current_username = None
            is_listening = False
            camera_enabled = False
            socketio.emit('camera_state', {'enabled': camera_enabled})
            socketio.emit('state_change', {'state': 'stopped'})
            break
        else:
            voice_system.speak("I didn't understand that command. Please try again.")

def handle_voice_input():
    """Main voice input handling loop"""
    global is_listening, current_state, voice_system
    
    while is_listening:
        try:
            if current_state == "welcome":
                handle_username_flow()
            elif current_state == "main_menu":
                handle_main_menu()
            else:
                time.sleep(0.2)
        except Exception as e:
            logging.error(f"Error in voice input handler: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

@app.route('/')
def index():
    """Render the modern main page"""
    return render_template('index2.html')

@app.route('/classic')
def index_classic():
    """Render the classic page"""
    return render_template('index.html')

def generate_video_frames():
    """Stream device camera frames with MediaPipe gesture overlay."""
    global camera_enabled, current_state
    camera_index = int(os.getenv("VOICE_EMAIL_CAM_INDEX", "0"))
    cap = None

    def build_status_frame(message):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buffer.tobytes()

    mp_hands = None
    hands = None
    mp_draw = None
    mp_styles = None
    last_gesture = None
    last_emit_ts = 0.0

    try:
        import mediapipe as mp_mediapipe
        if hasattr(mp_mediapipe, "solutions"):
            mp_hands = mp_mediapipe.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.2
            )
            mp_draw = mp_mediapipe.solutions.drawing_utils
            mp_styles = mp_mediapipe.solutions.drawing_styles
        else:
            socketio.emit("error", {"message": "MediaPipe does not expose solutions API in this environment."})
    except Exception as exc:
        socketio.emit("error", {"message": f"MediaPipe init failed: {exc}"})

    try:
        while True:
            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                jpg = build_status_frame("Camera is OFF (toggle to enable)")
                if jpg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(0.15)
                continue

            if cap is None:
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    jpg = build_status_frame("Camera unavailable. Check device/index.")
                    if jpg is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                        )
                    time.sleep(0.2)
                    continue

            with video_lock:
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            # Mirror for intuitive UI preview.
            frame = cv2.flip(frame, 1)

            # Run gesture recognition and annotate frame when MediaPipe is available.
            gesture = None
            if hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )
                        gesture = detect_gesture(hand_landmarks)

            if gesture:
                label = "THUMBS UP -> SEND" if gesture == "send" else "PALM -> CANCEL"
                cv2.rectangle(frame, (0, 0), (320, 34), (0, 0, 0), -1)
                cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                now = time.time()
                if gesture != last_gesture or (now - last_emit_ts) > 1.5:
                    socketio.emit("gesture", {"gesture": gesture})
                    last_emit_ts = now
                    last_gesture = gesture

                    # One-shot gesture action in compose mode:
                    # after first detected gesture command, turn camera off to avoid repeats.
                    if current_state == "compose":
                        camera_enabled = False
                        socketio.emit('camera_state', {'enabled': camera_enabled})

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            jpg = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
    finally:
        if hands is not None:
            hands.close()
        if cap is not None:
            cap.release()

@app.route('/video_feed')
def video_feed():
    """Device camera feed rendered inside browser UI."""
    return Response(
        generate_video_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global is_listening, voice_system, listening_thread, camera_enabled, connected_clients
    
    print("Client connected")
    with listener_lock:
        connected_clients += 1
    socketio.emit('camera_state', {'enabled': camera_enabled})
    
    # Initialize voice system if not already done
    if voice_system is None:
        try:
            # Needed for macOS
            mp.set_start_method('spawn', force=True)
            
            # Create necessary directories
            Path('logs').mkdir(exist_ok=True)
            Path('voice_auth').mkdir(exist_ok=True)
            
            # Initialize logging
            logging.basicConfig(
                filename='logs/web_app.log',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
            # Check for credentials
            if not Path('credentials.json').exists():
                error_msg = "credentials.json not found! Please add Gmail API credentials."
                print(f"\n[ERROR] {error_msg}")
                socketio.emit('error', {'message': error_msg})
                return
            
            # Create voice system instance
            voice_system = WebVoiceEmailSystem()
            
            # Verify microphone access
            if not voice_system.verify_microphone_access():
                socketio.emit('error', {'message': 'Microphone initialization failed'})
                return
            
            print("✅ Voice system initialized")
        except Exception as e:
            print(f"Error initializing voice system: {e}")
            socketio.emit('error', {'message': f'Initialization error: {e}'})
            return
    
    # Start listening thread if not already running
    with listener_lock:
        thread_alive = listening_thread is not None and listening_thread.is_alive()
        if not thread_alive:
            is_listening = True
            listening_thread = threading.Thread(target=handle_voice_input, daemon=True)
            listening_thread.start()
            print("✅ Voice input handler started")
            # Welcome message will be spoken in handle_username_flow() to avoid overlap

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_listening, connected_clients
    print("Client disconnected")
    with listener_lock:
        connected_clients = max(0, connected_clients - 1)
        # Stop backend loop only when last client disconnects.
        if connected_clients == 0:
            is_listening = False

@socketio.on('voice_command')
def handle_voice_command(data):
    """Handle manual voice commands from frontend (optional)"""
    global mic_pause_until
    raw = (data or {}).get("command", "")
    command = str(raw).strip().lower()
    if not command:
        return

    mic_pause_until = time.time() + float(os.getenv("VOICE_EMAIL_MANUAL_PAUSE_SEC", "3.0"))
    with manual_command_lock:
        manual_command_queue.append(command)
    print(f"Queued manual command: {command}")

@socketio.on('text_input')
def handle_text_input(data):
    """Handle typed user replies (email/subject/body/queries/confirmations)."""
    global mic_pause_until
    raw = (data or {}).get("text", "")
    text = str(raw).strip()
    if not text:
        return

    mic_pause_until = time.time() + float(os.getenv("VOICE_EMAIL_MANUAL_PAUSE_SEC", "3.0"))
    with typed_input_lock:
        typed_input_queue.append(text)
    print(f"Queued typed input: {text}")

@socketio.on('set_camera_enabled')
def handle_set_camera_enabled(data):
    """Toggle camera stream state from frontend."""
    global camera_enabled
    enabled = bool((data or {}).get('enabled', True))
    camera_enabled = enabled
    socketio.emit('camera_state', {'enabled': camera_enabled})

if __name__ == '__main__':
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('voice_auth').mkdir(exist_ok=True)
    
    # Initialize logging
    logging.basicConfig(
        filename='logs/web_app.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n🌐 Server starting...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🔊 Make sure your microphone is connected and permissions are granted\n")
    
    # Start the application
    try:
        socketio.run(app, debug=False, port=5001, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        is_listening = False
