# Voice Mail Setup Guide (Windows)

This guide captures the known-good setup from chat history, including the exact version pins that resolved runtime conflicts.

## 1) Create and activate virtual environment

```powershell
cd "D:\projects\anni_ka_project\Speak send\voice-mail\voice-mail"
deactivate
Remove-Item -Recurse -Force .\venv
py -m venv venv
.\venv\Scripts\Activate.ps1
python -V
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

If `py` is unavailable:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 2) Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install base libraries

```powershell
python -m pip install SpeechRecognition pyaudio Flask Flask-SocketIO opencv-python scipy scikit-learn google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client python-dotenv librosa sounddevice flask-wtf python-socketio python-engineio matplotlib seaborn pyttsx3
```

## 4) Apply known-good compatibility pins (required)

These are the versions that resolved the SpeechBrain/HuggingFace/Torch conflicts.

```powershell
python -m pip uninstall -y speechbrain hyperpyyaml huggingface_hub transformers sentence-transformers tokenizers torch torchaudio numpy mediapipe
python -m pip install "speechbrain==0.5.16" "hyperpyyaml==1.2.2" "huggingface_hub==0.24.7" "transformers==4.41.2" "sentence-transformers==2.7.0" "tokenizers==0.19.1" "torch==2.2.2" "torchaudio==2.2.2" "numpy==1.26.4" "mediapipe==0.10.13"
```

## 5) Verify environment

```powershell
python -c "import numpy, torch, torchaudio; print(numpy.__version__, torch.__version__, torchaudio.__version__, hasattr(torchaudio,'set_audio_backend'))"
python -c "import speechbrain, transformers, sentence_transformers, huggingface_hub; print(speechbrain.__version__, transformers.__version__, sentence_transformers.__version__, huggingface_hub.__version__)"
python -c "import pyaudio, mediapipe as mp; print('pyaudio ok', mp.__version__)"
```

Expected:
- NumPy `1.26.4`
- Torch/Torchaudio `2.2.2`
- `hasattr(torchaudio, 'set_audio_backend')` -> `True`

## 6) Run app (web UI)

Preferred:

```powershell
python web_app.py
```

Alternative launcher:

```powershell
python app.py
```

CLI mode:

```powershell
python app.py --mode cli
```

## 7) Gmail OAuth/auth reset

If token is stale or you do not get the auth flow:

```powershell
Remove-Item ".\token.pickle" -Force -ErrorAction SilentlyContinue
python -c "from gmail_service import get_service; get_service(); print('gmail auth ok')"
```

## 8) Common runtime fixes

### Camera not turning on

```powershell
$env:VOICE_EMAIL_CAM_INDEX="0"
python web_app.py
```

If needed, try index `1`, `2`, etc.

### Wrong microphone chosen

```powershell
$env:VOICE_EMAIL_MIC_INDEX="1"
python web_app.py
```

or

```powershell
$env:VOICE_EMAIL_MIC_NAME="Microphone Array"
python web_app.py
```
