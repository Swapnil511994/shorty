import os
import time
import pandas as pd
import re
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# === CONFIG ===
DEFAULT_QUEUE_CSV = os.getenv("UPLOAD_QUEUE_PATH", "pipelines/upload_queue.csv")
CREDENTIALS_FILE = "client_secrets.json"
TOKEN_FILE = "token.pickle"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
REQUIRED_COLUMNS = ["Index", "VideoPath", "Title", "Description", "Tags", "CategoryID", "UploadStatus", "YouTubeVideoID", "YouTubeVideoUrl"]

# === AUTH ===
def authenticate_youtube():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)
    return build("youtube", "v3", credentials=creds)

# === Upload Video ===
def upload_video(youtube, video_path, title, description, tags, category_id):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": [t.strip().lower() for t in tags.split(",") if t.strip()],
            "categoryId": str(category_id),
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, resumable=True, chunksize=1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    total_size = os.path.getsize(video_path)
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(video_path)}")

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                uploaded = int(status.resumable_progress)
                pbar.update(uploaded - pbar.n)
        except HttpError as e:
            print(f"❌ Upload failed: {e}")
            return None
    pbar.close()
    return response.get("id")

# === Ensure CSV Columns Exist ===
def ensure_queue_csv(queue_csv):
    if not os.path.exists(queue_csv):
        os.makedirs(os.path.dirname(queue_csv), exist_ok=True)
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(queue_csv, index=False)
    else:
        df = pd.read_csv(queue_csv)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df.to_csv(queue_csv, index=False)

# === Upload Loop ===
def run_upload_loop(queue_csv):
    print(f"📂 Watching queue: {queue_csv}")
    youtube = authenticate_youtube()

    while True:
        ensure_queue_csv(queue_csv)
        df = pd.read_csv(queue_csv)

        for idx, row in df.iterrows():
            if str(row.get("UploadStatus", "")).lower() == "completed":
                continue

            if str(row.get("UploadStatus", "")).lower() == "failed":
                continue

            if str(row.get("UploadStatus", "")).lower().startswith("error"):
                continue

            video_path = str(row.get("VideoPath", "")).strip()
            title = str(row.get("Title", "")).strip()
            desc = str(row.get("Description", "")).strip()
            tags = str(row.get("Tags", "")).strip()
            category_id = str(row.get("CategoryId", "")).strip()
            # print(f"🔍 Checking Category Id: {category_id}...")

            if not all([os.path.exists(video_path), title, desc, tags, category_id]):
                print(f"⚠️ Skipping Index {row.get('Index', idx)}: Incomplete metadata or missing file")
                continue

            print(f"\n🚀 Uploading Index {row.get('Index', idx)}...")

            try:
                video_id = upload_video(youtube, video_path, title, desc, tags, category_id)
                if video_id:
                    youtube_url = f"https://youtu.be/{video_id}"
                    df.at[idx, "YouTubeVideoID"] = video_id
                    df.at[idx, "YouTubeVideoUrl"] = youtube_url
                    df.at[idx, "UploadStatus"] = "completed"
                    print(f"✅ Uploaded: {youtube_url}")
                else:
                    df.at[idx, "UploadStatus"] = "failed"
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                df.at[idx, "UploadStatus"] = f"failed: {str(e)}"

            time.sleep(5)

        df.to_csv(queue_csv, index=False)
        print("⏳ Sleeping before next scan...\n")
        time.sleep(30)  # Sleep 30 seconds before checking again

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Upload Daemon")
    parser.add_argument("--queue", type=str, help="Path to upload queue CSV", default=DEFAULT_QUEUE_CSV)
    args = parser.parse_args()

    run_upload_loop(args.queue)
