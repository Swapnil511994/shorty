import os
import argparse
import glob
import shutil

# Mapping flags to folders
DELETE_PATHS = {
    "stories": "stories/generated",
    "metadata": "stories/metadata",
    "audio": "audio/narrations",
    "videos": "output/final_videos",
    "subtitles": "subtitles/srt",
}

def delete_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return

    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
        except Exception as e:
            print(f"❌ Failed to delete {file}: {e}")

    print(f"✅ Deleted all contents in: {folder_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean up pipeline-generated files")
    parser.add_argument("--delete-stories", action="store_true", help="Delete stories")
    parser.add_argument("--delete-metadata", action="store_true", help="Delete metadata")
    parser.add_argument("--delete-audio", action="store_true", help="Delete audio narrations")
    parser.add_argument("--delete-videos", action="store_true", help="Delete final videos")

    args = parser.parse_args()

    if args.delete_stories:
        delete_files(DELETE_PATHS["stories"])
    if args.delete_metadata:
        delete_files(DELETE_PATHS["metadata"])
    if args.delete_audio:
        delete_files(DELETE_PATHS["audio"])
        delete_files(DELETE_PATHS["subtitles"])
    if args.delete_videos:
        delete_files(DELETE_PATHS["videos"])

    if not any(vars(args).values()):
        print("ℹ️  No flags passed. Use --help to see options.")

if __name__ == "__main__":
    main()
