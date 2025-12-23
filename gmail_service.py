import os
import pickle
import base64
import re
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from pathlib import Path

SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

class GmailServiceError(Exception):
    """Custom exception for Gmail service errors"""
    pass

def validate_email(email):
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def get_service():
    """Get authenticated Gmail service with improved error handling"""
    try:
        creds = None
        token_path = Path('token.pickle')
        credentials_path = Path('credentials.json')

        # Check if credentials.json exists
        if not credentials_path.exists():
            raise GmailServiceError("credentials.json not found. Please set up OAuth 2.0 credentials.")

        # Load token from file if it exists
        if token_path.exists():
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)

        # Refresh or create if not valid
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    token_path.unlink(missing_ok=True)  # Delete invalid token
                    raise GmailServiceError("Failed to refresh credentials. Please authenticate again.")
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
                    creds = flow.run_local_server(port=0)  # Use random available port
                except Exception as e:
                    raise GmailServiceError(f"Authentication failed: {e}")

            # Save the credentials
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

        return build('gmail', 'v1', credentials=creds)

    except Exception as e:
        raise GmailServiceError(f"Failed to initialize Gmail service: {e}")

def send_email_via_gmail(to, subject, body):
    """Send email with improved validation and error handling"""
    try:
        # Validate inputs
        if not all([to, subject, body]):
            raise ValueError("Email, subject, and body are required")
        
        if not validate_email(to):
            raise ValueError(f"Invalid email address: {to}")
        
        # Get Gmail service
        service = get_service()
        
        # Create message
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        # Encode and send message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_result = service.users().messages().send(
            userId='me', 
            body={'raw': raw_message}
        ).execute()
        
        print(f"📤 Email sent successfully: Message ID {send_result['id']}")
        return True
        
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        raise
    except HttpError as e:
        print(f"❌ Gmail API error: {e}")
        raise GmailServiceError(f"Failed to send email: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise GmailServiceError(f"Failed to send email: {e}")

def get_most_recent_email():
    """Get the most recent email with error handling"""
    try:
        service = get_service()
        
        # Get the most recent email
        results = service.users().messages().list(
            userId='me',
            maxResults=1
        ).execute()
        
        if 'messages' not in results:
            return None
            
        msg_id = results['messages'][0]['id']
        message = service.users().messages().get(
            userId='me',
            id=msg_id,
            format='full'
        ).execute()
        
        # Extract email details
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No subject')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown sender')
        
        # Get email body
        if 'parts' in message['payload']:
            body = message['payload']['parts'][0]['body'].get('data', '')
        else:
            body = message['payload']['body'].get('data', '')
            
        if body:
            body = base64.urlsafe_b64decode(body).decode()
        else:
            body = 'No content'
            
        email_content = {
            'from': sender,
            'subject': subject,
            'body': body
        }
        
        return f"From: {sender}\nSubject: {subject}\nBody: {body}", email_content
        
    except Exception as e:
        print(f"❌ Error fetching recent email: {e}")
        raise GmailServiceError(f"Failed to fetch recent email: {e}")
