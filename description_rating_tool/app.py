import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials


SERVICE_ACCOUNT_FILE = "regrets-reporter-dev-d4b0ee4d637f.json"

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# The ID of your public-writeable Google Sheet
SHEET_ID = '1idnaMs-9k7adGO1kIOeu5wR8wwkkmclQ7LjF8y4NAZE'

TT_URL_FORMAT = "https://www.tiktok.com/@doesnotmatter/video/{vid}"

# Authenticate and create the client
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# Open the Google Sheet by ID
sheet = client.open_by_key(SHEET_ID).sheet1

# Get all records of the data
data = sheet.get_all_records()

# Convert data into DataFrame
data_df = pd.DataFrame(data)

# Get unique values in the 'video_path' column
unique_video_paths = data_df['video_path'].unique()

# Choose one randomly
random_video_path = np.random.choice(unique_video_paths)
video_id = random_video_path.split('/')[-1].split('.')[0]

st.set_page_config(layout="wide")
st.title("Description Eval Tool")

# Print the randomly chosen video path to the page
st.write(f"TikTok to evaluate: {TT_URL_FORMAT.format(vid=video_id)}")

