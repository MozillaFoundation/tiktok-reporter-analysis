import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from lingua import Language, LanguageDetectorBuilder


def load_descriptions(descriptions_path, sheet_id, current_model):
    language_detector = LanguageDetectorBuilder.from_all_languages().build()
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    # Authenticate and create the client
    creds = Credentials.from_service_account_file(
        "tiktok_reporter_analysis/regrets-reporter-dev-d4b0ee4d637f.json", scopes=SCOPES
    )
    client = gspread.authorize(creds)

    # Open the Google Sheet by ID
    sheet = client.open_by_key(sheet_id).worksheet("descriptions")
    df = pd.read_parquet(descriptions_path)

    # Extract the 'video_path' and 'description' fields
    data_to_append = [
        [row["video_path"], row["description"], row["audio_transcript"]]
        for _, row in df.iterrows()
        if row["audio_transcript"] == "" or language_detector.detect_language_of(row["audio_transcript"]) == Language.ENGLISH
    ]

    # Append the data to the Google Sheet
    for row in data_to_append:
        row.append(current_model)
    sheet.append_rows(data_to_append)

    print("Data appended successfully!")
