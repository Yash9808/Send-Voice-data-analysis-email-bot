import os
import ftplib
import assemblyai as aai
import pandas as pd
from pydub import AudioSegment
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import json
import base64
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText  # Import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# FTP Server details
HOST = "cph.v4one.co.uk"
USERNAME = "username" 
PASSWORD = "password" 

# AssemblyAI API Key
aai.settings.api_key = apikey 

# Load sensitive data from environment variables

ASSEMBLYAI_API_KEY = aai.settings.api_key

aai.settings.api_key = ASSEMBLYAI_API_KEY

# Function to list available folders
def list_folders(ftp):
    ftp.cwd('/')
    folders = [f for f in ftp.nlst() if '.' not in f]  # Assuming folders don't have file extensions
    return folders

# Function to download MP3 files from FTP
def download_mp3_files(ftp, remote_folder, local_folder):
    try:
        os.makedirs(local_folder, exist_ok=True)
        ftp.cwd(remote_folder)
        files = ftp.nlst()
        mp3_files = [file for file in files if file.endswith('.mp3')]

        for file in mp3_files:
            local_file_path = os.path.join(local_folder, os.path.basename(file))
            with open(local_file_path, "wb") as local_file:
                ftp.retrbinary(f"RETR {file}", local_file.write)
            print(f"Downloaded: {file}")
        return mp3_files
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to check if audio is valid
def check_audio_validity(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        if len(audio.get_array_of_samples()) == 0:
            return False
        return True
    except Exception:
        return False

# Sentiment and Tone analysis for agent and customer
def analyze_sentiment_and_tone(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    apw = text.lower().count("good")  # Example: Counting positive words
    anw = text.lower().count("bad")  # Number of negative words
    tone = analyze_tone(text)
    return sentiment_score, apw, anw, tone

def analyze_tone(text):
    sentiment = TextBlob(text).sentiment
    if sentiment.polarity > 0.1:
        tone = "Positive"
        tone_score = sentiment.polarity
    elif sentiment.polarity < -0.1:
        tone = "Negative"
        tone_score = sentiment.polarity
    else:
        tone = "Neutral"
        tone_score = 0.5  # Neutral tone
    return tone, tone_score

# Transcribe audio using AssemblyAI with retry logic
def transcribe_audio_with_retry(audio_file, retries=3, delay=5):
    for attempt in range(retries):
        try:
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_file)
            if transcript.status == aai.TranscriptStatus.completed:
                return transcript.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    return ""  # Return empty if transcription failed

# Process each audio file and save results
def process_audio_file(mp3_file, local_folder, i):
    audio_path = os.path.join(local_folder, mp3_file)
    
    # Convert MP3 to WAV (16kHz, Mono)
    audio = AudioSegment.from_mp3(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)[:30000]  # 30s limit
    wav_file = f"{local_folder}/temp_{i}.wav"
    audio.export(wav_file, format="wav")

    results = []
    if check_audio_validity(wav_file):
        text = transcribe_audio_with_retry(wav_file)
        
        if text:
            print(f"Transcription: {text}")

            # Sentiment and tone analysis
            agent_sentiment_score, apw, anw, agent_tone = analyze_sentiment_and_tone(text)
            customer_sentiment_score, cpw, cnw, customer_tone = analyze_sentiment_and_tone(text)

            # Compute Overall Score
            overall_score = (
                (0.3 * (agent_sentiment_score + 1) / 2) +
                (0.3 * (customer_sentiment_score + 1) / 2) +
                (0.2 * (agent_tone[1] + 1) / 2) +
                (0.2 * (apw - anw + cpw - cnw + 5) / 10)
            )
            overall_score = max(0, min(1, overall_score))

            results.append([  
                mp3_file, agent_tone[0], customer_tone[0], agent_tone[0], overall_score,
                agent_sentiment_score, customer_sentiment_score, agent_tone[1], apw, anw, cpw, cnw
            ])
        
        os.remove(wav_file)
    return results

# Initialize FTP connection
ftp = ftplib.FTP(HOST)
ftp.login(USERNAME, PASSWORD)

# List available folders
print("Available folders:", list_folders(ftp))

# Select folder
remote_folder = input("Enter the folder to process (e.g., 2025-03-03): ")
local_folder = f"download_audio/{remote_folder}"
os.makedirs(local_folder, exist_ok=True)

# Download files
mp3_files = download_mp3_files(ftp, remote_folder, local_folder)

# Process and analyze each file concurrently
results = []
with ThreadPoolExecutor() as executor:
    results = executor.map(lambda i: process_audio_file(mp3_files[i], local_folder, i), range(len(mp3_files)))

# Flatten results
results = [item for sublist in results for item in sublist]

# Save results to CSV
df = pd.DataFrame(results, columns=[  
    "File", "Agent Sentiment", "Customer Sentiment", "Agent Tone", "Overall Score",
    "Agent Sentiment Score", "Customer Sentiment Score", "Agent Tone Score", "APW", "ANW", "CPW", "CNW"
])
df.to_csv(f"{local_folder}/call_analysis_results.csv", index=False)
print("Analysis completed! Results saved to CSV.")

# Load the CSV
df_predictions = pd.read_csv(f"{local_folder}/call_analysis_results.csv")

# Drop unnecessary columns
df_predictions.drop(["File", "Agent Sentiment", "Customer Sentiment"], axis=1, inplace=True)

# Initialize LabelEncoder for 'Agent Tone'
le = LabelEncoder()
df_predictions["Agent Tone"] = le.fit_transform(df_predictions["Agent Tone"])

# Create a folder for saving the figures
folder_name = datetime.now().strftime("%d-%m-%y") or remote_folder
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 1. Plot: Distribution of 'Agent Tone'
plt.figure(figsize=(10, 6))
df_predictions["Agent Tone"].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Agent Tone')
plt.xlabel('Agent Tone')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.savefig(os.path.join(folder_name, 'Figure1.png'))
plt.show()
plt.close()

# 2. Plot: Scatter plot of 'Agent Sentiment Score' vs 'Customer Sentiment Score'
plt.figure(figsize=(10, 6))
plt.scatter(df_predictions['Agent Sentiment Score'], df_predictions['Customer Sentiment Score'], color='orange')
plt.title('Agent Sentiment Score vs Customer Sentiment Score')
plt.xlabel('Agent Sentiment Score')
plt.ylabel('Customer Sentiment Score')
plt.grid(True)
plt.savefig(os.path.join(folder_name, 'Figure2.png'))
plt.show()
plt.close()

# 3. Plot: Scatter plot of 'Agent Tone Score' vs 'Overall Score'
plt.figure(figsize=(10, 6))
plt.scatter(df_predictions['Agent Tone Score'], df_predictions['Overall Score'], color='green')
plt.title('Agent Tone Score vs Overall Score')
plt.xlabel('Agent Tone Score')
plt.ylabel('Overall Score')
plt.grid(True)
plt.savefig(os.path.join(folder_name, 'Figure3.png'))
plt.show()
plt.close()

# 4. Plot: Correlation heatmap to visualize relationships between numerical variables
plt.figure(figsize=(10, 6))
correlation_matrix = df_predictions[['Agent Sentiment Score', 'Customer Sentiment Score', 'Agent Tone Score', 'Overall Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(folder_name, 'Figure4.png'))
plt.show()
plt.close()

# 5. Plot: Sample index vs Overall Score
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(df_predictions)), df_predictions['Overall Score'], marker='o', color='b', linestyle='-', markersize=5)
plt.title('Sample Index vs Overall Score')
plt.xlabel('Sample Index')
plt.ylabel('Overall Score')
plt.grid(True)
plt.savefig(os.path.join(folder_name, 'Figure5.png'))
plt.show()
plt.close()

# Clean up - delete the audio files after analysis
for file in mp3_files:
    os.remove(os.path.join(local_folder, file))

ftp.quit()

#############################################send emial###################################################
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Check if token.json exists and load credentials
creds = None
if os.path.exists('token.json'):
    try:
        with open('token.json', 'r') as token_file:
            creds_data = json.load(token_file)
        creds = Credentials.from_authorized_user_info(creds_data, SCOPES)

        # Check if credentials are valid
        if creds and creds.valid:
            print("Credentials are valid.")
        else:
            print("Credentials are not valid.")
    except ValueError as e:
        print(f"❌ Error loading credentials: {e}")
else:
    print("Token file not found.")

# Build the Gmail service using the credentials
service = build('gmail', 'v1', credentials=creds)

# Define email content
sender = "ysahsharmaysa@gmail.com"
recipients = ['email1', 'email2','email3'] 
subject = "Test Bot Email with Attachment"
body = "Hi, this is an automated email from a bot with an attachment of Data analysis achieved from Booking voice call. It shows Agent Tone, Overall Score, Agent Sentiment Score, Customer Sentiment Score, Agent Tone Score. For any details contact:yash.sharma@chartwellprivatehospital.co.uk "

# Prepare the recipients string as a comma-separated list
recipient_string = ", ".join(recipients)

# Create a MIMEMultipart email object
message = MIMEMultipart()
message['From'] = sender
message['To'] = recipient_string
message['Subject'] = subject

# Attach the body of the email
message.attach(MIMEText(body, 'plain'))

#######################################correct till here######################

# Path to the folder containing the images (adjust as needed)

folder_path = os.path.join(r"C:\Users\wsys3\Downloads\Telesales - Call recording\Auto_report_sender", remote_folder)

# Check if folder exists
if not os.path.exists(folder_path):
    print(f"❌ Error: Folder '{folder_path}' does not exist.")
    exit()

# Get image files
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("⚠️ No image files found. Exiting.")
    exit()

# Create email
message = MIMEMultipart()
message['From'] = sender
message['To'] = recipient_string
message['Subject'] = subject
message.attach(MIMEText(body, 'plain'))

# Attach images
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    try:
        with open(image_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={image_file}')
            message.attach(part)
            print(f"✅ Attached: {image_file}")
    except FileNotFoundError:
        print(f"❌ Error: {image_file} not found.")

# Send email only if attachments exist
if message.get_payload():
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    try:
        sent_message = service.users().messages().send(userId="me", body={'raw': raw_message}).execute()
        print(f"✅ Email sent! Message ID: {sent_message['id']}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
else:
    print("⚠️ No attachments found. Email not sent.")

