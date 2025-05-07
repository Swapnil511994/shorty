import argparse
import os
import time
import pandas as pd
import re
import pickle
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# === CONFIG ===
CSV_PATH = 'data/input.csv'
CREDENTIALS_FILE = 'client_secrets.json'
TOKEN_FILE = 'token.pickle'
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

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

# === Upload ===
def upload_video(youtube, video_path, title, description, tags, category_id):
    title = re.sub(r'"', '', title).strip() or "Untitled"
    description = re.sub(r'"', '', description).strip() or "No description."
    tags = [t.strip().lower() for t in tags.split(',') if t.strip()]

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": str(category_id)
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, resumable=True, chunksize=1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    total_size = os.path.getsize(video_path)
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Uploading", colour="green")
    response = None

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                uploaded = int(status.resumable_progress)
                pbar.update(uploaded - pbar.n)
        except HttpError as e:
            raise RuntimeError(f"YouTube API error: {e}")
    pbar.close()

    return response.get("id") if response else None

# === Main ===
def main(category_id):
    youtube = authenticate_youtube()
    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        if str(row.get("UploadStatus", "")).lower() == "completed":
            continue

        video_path = str(row.get("VideoPath", "")).strip()
        title = str(row.get("Title", "")).strip()
        desc = str(row.get("Description", "")).strip()
        tags = str(row.get("Tags", "")).strip()

        if not os.path.exists(video_path):
            print(f"⚠️ Skipping ID {row.get('ID', idx)}: video not found")
            df.at[idx, "UploadStatus"] = "failed: file not found"
            continue

        print(f"\n🚀 Uploading video ID {row.get('ID', idx)}...")

        try:
            video_id = upload_video(youtube, video_path, title, desc, tags, category_id)
            if video_id:
                youtube_url = f"https://youtu.be/{video_id}"
                df.at[idx, "YouTubeVideoID"] = video_id
                df.at[idx, "YouTubeVideoUrl"] = youtube_url
                df.at[idx, "UploadStatus"] = "completed"
                print(f"✅ Uploaded: {youtube_url}")
            else:
                df.at[idx, "UploadStatus"] = "failed: no response"
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            df.at[idx, "UploadStatus"] = f"failed: {str(e)}"

        time.sleep(5)

    df.to_csv(CSV_PATH, index=False)
    print("\n📦 All uploads complete. CSV updated.")

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload videos to YouTube with optional category")
    parser.add_argument('--category', type=str, default="22", help="YouTube category ID (default: 22 = People & Blogs)")
    args = parser.parse_args()

    main(args.category)
