from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import os
from pathlib import Path
import logging
import multiprocessing as mp

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

class WebVoiceEmailSystem(VoiceEmailSystem):
    """Wrapper around VoiceEmailSystem that uses say command AND emits to SocketIO for UI"""
    
    def speak(self, text):
        """Override speak to use say command (for blind users) AND emit to SocketIO (for teacher UI)"""
        try:
            print(f"[TTS] {text}")
            
            # Use say command for actual voice output (for blind users) - same as app.py
            import subprocess
            subprocess.run(['say', '-r', str(self.speech_rate), text], check=True)
            time.sleep(0.3)
            
            # ALSO emit to SocketIO for visual UI (for teachers to see what's happening)
            socketio.emit('speak', {'text': text})
        except Exception as e:
            print(f"Error in speak function: {e}")
    
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
                current_username = None
                return
    
    if not username:
        return
    
    current_username = username
    
    # Check if user exists
    profile_path = Path("voice_auth") / f"{username}_profile.pkl"
    
    if not profile_path.exists():
        # New user - enroll voice
        if not voice_system.handle_voice_enrollment(username):
            current_state = "welcome"
            current_username = None
            return
    else:
        # Existing user - verify voice
        voice_system.speak(f"Welcome back {username}! Let's verify your voice.")
        if not voice_system.authenticate_user(username):
            current_state = "welcome"
            current_username = None
            return
    
    # Success - go to main menu
    current_state = "main_menu"
    socketio.emit('state_change', {'state': 'main_menu'})

def handle_main_menu():
    """Handle main menu commands - matches app.py exactly"""
    global current_state, current_username, voice_system
    
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
            voice_system.handle_email_composition(current_username)
            current_state = "main_menu"
            socketio.emit('state_change', {'state': 'main_menu'})
        elif command == "search":
            current_state = "search"
            socketio.emit('state_change', {'state': 'search'})
            voice_system.handle_email_search()
            current_state = "main_menu"
            socketio.emit('state_change', {'state': 'main_menu'})
        elif command == "recent":
            current_state = "recent"
            socketio.emit('state_change', {'state': 'recent'})
            voice_system.handle_most_recent_email()
            current_state = "main_menu"
            socketio.emit('state_change', {'state': 'main_menu'})
        elif command == "exit":
            voice_system.speak("Goodbye!")
            current_state = "welcome"
            current_username = None
            socketio.emit('state_change', {'state': 'welcome'})
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
    """Render the main page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global is_listening, voice_system, listening_thread
    
    print("Client connected")
    
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
    if not is_listening:
        is_listening = True
        listening_thread = threading.Thread(target=handle_voice_input, daemon=True)
        listening_thread.start()
        print("✅ Voice input handler started")
        # Welcome message will be spoken in handle_username_flow() to avoid overlap

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_listening
    print("Client disconnected")
    is_listening = False

@socketio.on('voice_command')
def handle_voice_command(data):
    """Handle manual voice commands from frontend (optional)"""
    print(f"Received voice command: {data}")

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
