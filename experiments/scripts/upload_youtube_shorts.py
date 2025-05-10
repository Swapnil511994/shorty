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

# === Upload Video with Progress ===
def upload_video(youtube, video_path, title, description, tags):
    # Clean text
    title = re.sub(r'"', '', title).strip()
    description = re.sub(r'"', '', description).strip()
    tags = [t.strip().lower() for t in tags.split(',') if t.strip()]

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, resumable=True, chunksize=1024 * 1024)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    # Progress bar
    total_size = os.path.getsize(video_path)
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Uploading", colour="green")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            uploaded = int(status.resumable_progress)
            pbar.update(uploaded - pbar.n)
    pbar.close()

    return response.get("id") if response else None

# === MAIN ===
def main():
    youtube = authenticate_youtube()
    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        if str(row.get("UploadStatus", "")).lower() == "completed":
            continue

        video_path = row.get("VideoPath", "").strip()
        title = str(row.get("Title", "")).strip()
        desc = str(row.get("Description", "")).strip()
        tags = str(row.get("Tags", "")).strip()

        if not all([os.path.exists(video_path), title, desc, tags]):
            print(f"⚠️ Skipping ID {row.get('ID', idx)}: missing video, title, description or tags")
            continue

        print(f"\n🚀 Uploading video ID {row.get('ID', idx)}...")

        try:
            video_id = upload_video(youtube, video_path, title, desc, tags)
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

        time.sleep(5)  # prevent rate-limiting

    df.to_csv(CSV_PATH, index=False)
    print("\n📦 All uploads complete. CSV updated.")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()
