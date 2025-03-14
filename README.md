Audio Transcription and Sentiment Analysis Project
This project processes audio files from an FTP server, transcribes them, performs sentiment and tone analysis on the transcribed text, and generates visualizations. The analysis results are saved as CSV files and sent via email to specified recipients with attached plots.

Features:
FTP Integration: Downloads audio files from an FTP server.
Audio Conversion: Converts MP3 files to WAV format for processing.
Audio Transcription: Uses AssemblyAI API to transcribe audio into text.
Sentiment Analysis: Analyzes sentiment (positive/negative) and tone (agent/customer) of transcribed text using TextBlob.
Visualization: Generates visualizations (e.g., scatter plots, bar charts, heatmaps) to display analysis results.
Email Reports: Sends an email with the analysis results and visualizations as attachments.
Libraries Used:
assemblyai: For transcription of audio files.
pydub: For audio file format conversion (MP3 to WAV).
textblob: For sentiment and tone analysis.
numpy, matplotlib, seaborn: For data visualization.
scikit-learn: For label encoding.
google-api-python-client: For sending emails via Gmail.
ftplib: For FTP operations.
pandas: For data handling and CSV operations.
Prerequisites:
AssemblyAI API Key: You need an API key from AssemblyAI to transcribe the audio files. You can get it here.
Google API Credentials: The project uses the Gmail API to send emails. You'll need to create credentials by setting up a project on Google Cloud Console and enabling the Gmail API. Then, download the credentials JSON and save it as token.json.
FTP Server Details: You must specify the FTP server details (host, username, password) to connect and download the audio files.
Installation:
Clone the repository to your local machine.
Create a virtual environment:
bash
Copy
Edit
python -m venv venv
Activate the virtual environment:
On Windows:
bash
Copy
Edit
venv\Scripts\activate
On macOS/Linux:
bash
Copy
Edit
source venv/bin/activate
Install required packages:
bash
Copy
Edit
pip install -r requirements.txt
Configuration:
Set Up FTP: Ensure you have the correct FTP server details (hostname, username, and password).
AssemblyAI API Key: Set your AssemblyAI API key by replacing the placeholder in the code.
Google API Credentials: Create and download your Google API credentials and place them in your project directory as token.json.
Email Recipients: Modify the list of email recipients in the script to suit your needs.
Running the Script:
Make sure all dependencies are installed.

Run the script:

bash
Copy
Edit
python main.py
The script will:

Connect to the FTP server and list available folders.
Download all MP3 files from the selected folder.
Convert the MP3 files to WAV format and transcribe them.
Perform sentiment and tone analysis on the transcribed text.
Generate and save visualizations as PNG files.
Send an email with the results as a CSV file and the visualizations attached.
Example Output:
CSV File: The results of the analysis are saved in a CSV file with the following columns:

File: Audio file name
Agent Sentiment: Sentiment of the agent’s speech (Positive/Negative/Neutral)
Customer Sentiment: Sentiment of the customer’s speech (Positive/Negative/Neutral)
Agent Tone: Tone of the agent’s speech
Overall Score: Overall score based on sentiment and tone
Agent Sentiment Score: Numerical sentiment score of the agent’s speech
Customer Sentiment Score: Numerical sentiment score of the customer’s speech
Agent Tone Score: Numerical score representing the agent's tone
APW, ANW: Agent Positive/Negative word count
CPW, CNW: Customer Positive/Negative word count
Plots: The following plots are generated and saved:

Distribution of Agent Tone.
Agent Sentiment vs Customer Sentiment.
Agent Tone Score vs Overall Score.
Correlation heatmap of numerical variables.
Sample index vs Overall Score.
Example Email:
The email will contain:

A subject line: Test Bot Email with Attachment.
A body message with an explanation.
Attachments:
CSV file with the analysis results.
PNG files with visualizations (plots).
Troubleshooting:
Error connecting to FTP: Ensure that the FTP credentials and server details are correct.
API errors: Ensure the correct API keys and Google credentials are provided.
Missing files: Ensure the folder path is correct and the audio files are available.
![visual selection](https://github.com/user-attachments/assets/2d038cbc-6287-438d-a5a9-b660eab6a687)
