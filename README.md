# 📬 SpeakSend — Voice-Controlled Email System

A hands-free email client that lets you compose, send, search, and read emails using your **voice**, **gestures**, and a **browser-based UI**. Authentication is biometric — your voice is your password.

---

## ✨ Features

- 🎙️ **Voice Authentication** — Enroll and verify users via speaker recognition (SpeechBrain ECAPA-VOXCELEB model)
- ✉️ **Voice-Composed Emails** — Dictate recipient, subject, and body; send via Gmail API
- 🔍 **Semantic Email Search** — Natural language queries matched against your inbox using sentence embeddings
- 📥 **Recent Email Playback** — Read your latest emails aloud using text-to-speech
- 👋 **Gesture Control** — Thumbs up to send, open palm to cancel — detected live via webcam
- 🌐 **Web UI** — Browser-based interface powered by Flask + Socket.IO with real-time camera feed
- 💻 **CLI Mode** — Fully terminal-driven alternative with no browser required

---

## 🗂️ Project Structure

```
.
├── app.py                  # Main entry point; VoiceEmailSystem core logic (CLI + web launcher)
├── web_app.py              # Flask + Socket.IO web server; WebVoiceEmailSystem wrapper
├── voice_auth.py           # Speaker enrollment & verification (SpeechBrain)
├── gesture_process.py      # MediaPipe-based hand gesture detection (thumbs up / palm)
├── gmail_service.py        # Gmail API wrapper (send & read emails)
├── semantic_search.py      # Sentence-transformer email search + recent email fetcher
├── tts_trial.py            # Standalone TTS test script (pyttsx3)
├── test_camera.py          # Camera backend diagnostic tool
├── test_voice_auth.py      # Voice authentication test harness
├── generate_test_results.py# Test result generator
├── credentials.json        # Google OAuth 2.0 client credentials (see setup)
├── token.pickle            # Cached OAuth token (auto-generated after first login)
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- A working **microphone** and (optionally) a **webcam**
- A **Google account** with Gmail API access

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Some packages have native dependencies:
> - `pyaudio` requires PortAudio — on macOS: `brew install portaudio`; on Ubuntu: `sudo apt install portaudio19-dev`
> - `mediapipe` — install version `0.10.8` or `0.10.13` as listed in `requirements.txt`
> - `torch` / `torchaudio` — see [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific install commands

### 3. Set Up Gmail API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **Gmail API**
3. Create **OAuth 2.0 credentials** (Desktop App type)
4. Download the credentials and save as `credentials.json` in the project root
5. On first run, a browser window will open to authorise access — `token.pickle` is saved automatically for future sessions

### 4. Run the App

**Web mode** (default — opens browser UI at `http://localhost:5001`):
```bash
python app.py --mode web
```

**CLI mode** (terminal only):
```bash
python app.py --mode cli
```

---

## 🎮 How to Use

### Voice Commands (spoken or typed in the web chat box)

| Command    | Action                                      |
|------------|---------------------------------------------|
| `compose`  | Start composing a new email                 |
| `search`   | Search your inbox with a natural language query |
| `recent`   | Read the most recent email aloud            |
| `exit`     | Quit the application                        |
| `confirm`  | Confirm an action (e.g. send email)         |
| `cancel`   | Cancel current action                       |

### Gesture Controls (webcam required)

| Gesture      | Action        |
|--------------|---------------|
| 👍 Thumbs up | Send email    |
| ✋ Open palm | Cancel        |

### Voice Enrollment

On first use, you will be prompted to say your username and record **5 voice samples** using different phrases. Your voice profile is saved locally in the `voice_auth/` directory.

---

## ⚙️ Configuration (Environment Variables)

| Variable                        | Default | Description                                  |
|---------------------------------|---------|----------------------------------------------|
| `VOICE_EMAIL_MIC_INDEX`         | auto    | Force a specific microphone device index     |
| `VOICE_EMAIL_MIC_NAME`          | auto    | Select microphone by partial name match      |
| `VOICE_EMAIL_CALIBRATION_SEC`   | `0.8`   | Duration for ambient noise calibration       |
| `VOICE_EMAIL_LISTEN_TIMEOUT`    | `30`    | Max seconds to wait for speech input         |
| `VOICE_EMAIL_PHRASE_LIMIT`      | `20`    | Max phrase duration in seconds               |
| `VOICE_EMAIL_MIN_ENERGY`        | `50`    | Minimum energy threshold for recognition     |
| `VOICE_EMAIL_MAX_ENERGY`        | `1200`  | Maximum energy threshold for recognition     |
| `VOICE_EMAIL_MANUAL_PAUSE_SEC`  | `3.0`   | Mic pause after a typed/button command       |
| `VOICE_EMAIL_WEB_BACKEND_TTS`   | `0`     | Set to `1` to enable server-side TTS in web mode |

---

## 🛠️ Diagnostic Tools

```bash
# Test your camera and available backends
python test_camera.py

# Test text-to-speech
python tts_trial.py

# Run gesture recognition standalone
python gesture_process.py

# Run semantic email search directly
python semantic_search.py
```

---

## 🔒 Security Notes

- Voice profiles are stored locally as `.pkl` files in `voice_auth/` and never transmitted
- `credentials.json` and `token.pickle` contain sensitive OAuth data — **do not commit these to version control**
- Add both to your `.gitignore`:
  ```
  credentials.json
  token.pickle
  voice_auth/
  logs/
  ```

---

## 📦 Key Dependencies

| Package                  | Purpose                              |
|--------------------------|--------------------------------------|
| `flask` + `flask-socketio` | Web server and real-time communication |
| `SpeechRecognition`      | Microphone input and speech-to-text  |
| `speechbrain`            | Speaker embedding model (ECAPA)      |
| `mediapipe`              | Hand landmark detection for gestures |
| `opencv-python`          | Camera capture and frame processing  |
| `sentence-transformers`  | Semantic email search embeddings     |
| `google-api-python-client` | Gmail API integration              |
| `pyttsx3`                | Cross-platform text-to-speech (non-macOS) |
| `torch` / `torchaudio`   | Deep learning backend for voice auth |

---

## 🧭 Platform Notes

- **macOS**: Uses the built-in `say` command for TTS. Camera access requires permission in *System Preferences → Security & Privacy → Camera*.
- **Windows**: Uses `pyttsx3` for TTS. Set `VOICE_EMAIL_WEB_BACKEND_TTS=1` if you want server-side speech in web mode.
- **Linux**: Ensure `portaudio` and a working audio/camera stack are available.
