# Voice-Based Email System

A sophisticated email system that combines voice authentication, speech recognition, and gesture control to provide a hands-free email experience.

## Features

- **Voice Authentication**: Secure access using voice biometric features
- **Speech Recognition**: Natural language email composition and commands
- **Gesture Control**: Intuitive gesture-based commands (thumbs up to send, palm to cancel)
- **Gmail Integration**: Seamless integration with Gmail for sending and searching emails
- **Enhanced Security**: Multi-factor voice verification using pitch, formants, and tempo

## Prerequisites

- Python 3.8 or higher
- macOS (for the default text-to-speech engine)
- Microphone
- Webcam (for gesture control)
- Gmail account with API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-email-system
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Set up Gmail API credentials:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project and enable Gmail API
   - Create OAuth 2.0 credentials
   - Download and save as `credentials.json` in the project directory

## Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Run the application:
```bash
voice-email
```

3. Follow the voice prompts to:
   - Authenticate using your voice
   - Compose emails using speech
   - Use gestures to confirm or cancel actions
   - Search emails using voice commands

## Voice Commands

- "Send email" - Start composing a new email
- "Search email" - Search through your emails
- "Recent email" - Get your most recent email
- "Yes/No" - Confirm or cancel actions
- "Exit" - Close the application

## Gestures

- 👍 Thumbs Up - Confirm/Send email
- ✋ Palm - Cancel/Reject action

## Configuration

Edit the `.env` file to customize:
- Voice authentication settings
- Gesture recognition parameters
- Logging preferences
- API paths and tokens

## Troubleshooting

1. Voice Authentication Issues:
   - Ensure you're in a quiet environment
   - Speak clearly during enrollment
   - Try re-enrolling if recognition fails consistently

2. Gesture Recognition Issues:
   - Ensure good lighting
   - Keep your hand within camera view
   - Maintain proper distance from camera

3. Audio Issues:
   - Check microphone connections
   - Verify microphone permissions
   - Adjust system audio settings

## Logs

Logs are stored in `logs/app.log`. Check this file for detailed debugging information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Cloud Platform for Gmail API
- MediaPipe for gesture recognition
- librosa for audio processing
- OpenCV for computer vision 