import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
import base64
import email
from email.mime.text import MIMEText

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels'
]

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def get_email_content(service, msg_id):
    """Get the full content of an email"""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        
        # Get headers
        headers = message['payload']['headers']
        subject = next(h['value'] for h in headers if h['name'].lower() == 'subject')
        from_email = next(h['value'] for h in headers if h['name'].lower() == 'from')
        date = next(h['value'] for h in headers if h['name'].lower() == 'date')
        
        # Get body
        if 'parts' in message['payload']:
            parts = message['payload']['parts']
            body = ''
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    break
        else:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode()
            
        return {
            'id': msg_id,
            'subject': subject,
            'from': from_email,
            'date': date,
            'body': body
        }
    except Exception as e:
        print(f"Error getting email content: {e}")
        return None

def get_recent_emails(service, max_results=10):
    """Get recent emails"""
    try:
        results = service.users().messages().list(userId='me', maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        emails = []
        for message in messages:
            email_content = get_email_content(service, message['id'])
            if email_content:
                emails.append(email_content)
        
        return emails
    except Exception as e:
        print(f"Error getting recent emails: {e}")
        return []

def get_most_recent_email(service=None, offset=0):
    """Get and format a recent email by index offset (0 = newest)."""
    if not service:
        service = get_gmail_service()
    
    try:
        safe_offset = max(0, int(offset))
        # Fetch enough messages so we can select nth recent message.
        results = service.users().messages().list(userId='me', maxResults=safe_offset + 1).execute()
        messages = results.get('messages', [])
        
        if not messages or safe_offset >= len(messages):
            return None
            
        email_content = get_email_content(service, messages[safe_offset]['id'])
        if email_content:
            formatted_email = (
                f"From: {email_content['from']}\n"
                f"Date: {email_content['date']}\n"
                f"Subject: {email_content['subject']}\n"
                f"Message:\n{email_content['body']}"
            )
            return formatted_email, email_content
        return None
    except Exception as e:
        print(f"Error getting most recent email: {e}")
        return None

def run_semantic_search(query=None):
    print("🔄 Initializing semantic search...")
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    
    # Get Gmail service
    service = get_gmail_service()
    print("✅ Gmail service initialized")
    
    # Get recent emails
    print("\n📥 Fetching recent emails...")
    emails = get_recent_emails(service)
    
    if not emails:
        print("❌ No emails found")
        return None
    
    if not query:
        print("🎙️ Say your search query...")
        return None
    
    print(f"\n🔍 Searching for: {query}")
    
    # Create embeddings for the query and emails
    query_embedding = model.encode([query])[0]
    email_texts = [f"Subject: {email['subject']}\nBody: {email['body']}" for email in emails]
    email_embeddings = model.encode(email_texts)
    
    # Calculate similarities
    similarities = [np.dot(query_embedding, email_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(email_emb))
                   for email_emb in email_embeddings]
    
    # Sort emails by similarity
    sorted_results = sorted(zip(similarities, emails), key=lambda x: x[0], reverse=True)
    
    print("\n📬 Matching Emails:\n")
    found_matches = False
    matching_emails = []
    
    for similarity, email in sorted_results:
        if similarity > 0.3:  # Similarity threshold
            found_matches = True
            print(f"📧 From: {email['from']}")
            print(f"📅 Date: {email['date']}")
            print(f"📝 Subject: {email['subject']}")
            print(f"💌 Content: {email['body'][:200]}...")  # Show first 200 chars
            print(f"Relevance: {similarity:.2%}")
            print("\n" + "="*50 + "\n")
            
            matching_emails.append({
                'similarity': similarity,
                'email': email
            })
    
    if not found_matches:
        print("❌ No relevant emails found for your query.")
        return None
    
    return matching_emails

if __name__ == "__main__":
    # Test the semantic search
    run_semantic_search("Show me emails about internship offers")
