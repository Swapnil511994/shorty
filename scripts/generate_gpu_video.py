import os
import sys
import random
import subprocess
import pandas as pd
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self):
        self.config = {
            'CSV_PATH': 'data/input.csv',
            'VIDEO_OUTPUT_DIR': 'output/final_videos',
            'STOCK_VIDEO_DIR': 'video/backgrounds',
            'AUDIO_DIR': 'audio/narrations',
            'SUBTITLE_DIR': 'subtitles/srt',
            'MAX_WORKERS': 2,
            'FFMPEG_PATH': self.find_ffmpeg(),
            'FFPROBE_PATH': self.find_ffprobe()
        }
        self.validate_environment()

    def find_ffmpeg(self):
        """Locate ffmpeg binary with proper error handling"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  check=True)
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass
        
        # Check common installation paths
        paths = [
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('SYSTEMDRIVE', 'C:'), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg'
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
                
        raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")

    def find_ffprobe(self):
        """Locate ffprobe binary with proper error handling"""
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  check=True)
            if result.returncode == 0:
                return 'ffprobe'
        except:
            pass
        
        # Check common installation paths
        paths = [
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'ffmpeg', 'bin', 'ffprobe.exe'),
            os.path.join(os.environ.get('SYSTEMDRIVE', 'C:'), 'ffmpeg', 'bin', 'ffprobe.exe'),
            '/usr/bin/ffprobe',
            '/usr/local/bin/ffprobe'
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
                
        raise Exception("FFprobe not found. Please install FFmpeg and ensure it's in your PATH")

    def validate_environment(self):
        """Validate all required components exist"""
        required_dirs = ['STOCK_VIDEO_DIR', 'AUDIO_DIR', 'SUBTITLE_DIR']
        for dir_key in required_dirs:
            path = self.config[dir_key]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")
            if not os.listdir(path):
                raise FileNotFoundError(f"Directory is empty: {path}")
                
        os.makedirs(self.config['VIDEO_OUTPUT_DIR'], exist_ok=True)
        
        # Verify FFmpeg works
        try:
            result = subprocess.run(
                [self.config['FFMPEG_PATH'], '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            logger.info(f"Using FFmpeg version: {result.stdout.splitlines()[0]}")
        except Exception as e:
            raise Exception(f"FFmpeg verification failed: {str(e)}")

    def get_media_duration(self, file_path):
        """Get duration using ffprobe with comprehensive error handling"""
        try:
            result = subprocess.run(
                [self.config['FFPROBE_PATH'], '-v', 'error', '-show_entries', 
                 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=10  # 10 second timeout
            )
            return float(result.stdout.strip())
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting duration for {file_path}")
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting duration for {file_path}: {e.stderr}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error with {file_path}: {str(e)}")
            return 0

    def build_ffmpeg_command(self, video_paths, audio_path, subtitle_path, output_path):
        """Construct a dynamic FFmpeg command with subtitle rendering inside filter_complex"""
        # Convert paths to forward slashes for FFmpeg compatibility
        video_inputs = []
        for v in video_paths:
            video_inputs.extend(['-i', os.path.normpath(v).replace('\\', '/')])

        norm_audio = os.path.normpath(audio_path).replace('\\', '/')
        norm_subtitle = os.path.normpath(subtitle_path).replace('\\', '/')
        norm_output = os.path.normpath(output_path).replace('\\', '/')

        # Add video and audio inputs
        command = [self.config['FFMPEG_PATH'], '-y', *video_inputs, '-i', norm_audio]

        # Build filter_complex: concat + subtitles
        video_count = len(video_paths)
        concat_inputs = ''.join([f'[{i}:v]' for i in range(video_count)])
        filter_complex = (
            f"{concat_inputs}concat=n={video_count}:v=1:a=0"
            f",subtitles='{norm_subtitle}':force_style='Fontsize=24,Alignment=10'[v]"
        )

        command.extend([
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', f'{video_count}:a',  # The audio input is after all video inputs
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-shortest',
            norm_output
        ])

        return command

    
    def process_video(self, video_id):
        """Process a single video with comprehensive error handling"""
        try:
            # Prepare paths
            audio_path = os.path.join(self.config['AUDIO_DIR'], f'story_{video_id}.wav')
            subtitle_path = os.path.join(self.config['SUBTITLE_DIR'], f'story_{video_id}.srt')
            output_path = os.path.join(self.config['VIDEO_OUTPUT_DIR'], f'video_{video_id}.mp4')
            
            # Validate inputs
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file missing: {audio_path}")
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"Subtitle file missing: {subtitle_path}")
            
            # Get stock videos
            stock_videos = [
                os.path.join(self.config['STOCK_VIDEO_DIR'], f)
                for f in os.listdir(self.config['STOCK_VIDEO_DIR'])
                if f.lower().endswith(('.mp4', '.mov'))
            ]
            if not stock_videos:
                raise ValueError("No stock videos found")
            
            # Shuffle and select videos
            random.shuffle(stock_videos)
            
            # Calculate required duration
            audio_duration = self.get_media_duration(audio_path)
            if audio_duration <= 0:
                raise ValueError(f"Invalid audio duration: {audio_duration}")
            
            selected_videos = []
            total_duration = 0
            
            for video in stock_videos:
                dur = self.get_media_duration(video)
                if dur > 0:
                    selected_videos.append(video)
                    total_duration += dur
                    if total_duration >= audio_duration * 1.5:  # 50% buffer
                        break
            
            if not selected_videos:
                raise ValueError("No valid stock videos selected")
            
            # Build and run command
            command = self.build_ffmpeg_command(selected_videos, audio_path, subtitle_path, output_path)
            logger.debug(f"FFmpeg command for ID {video_id}: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                if process.returncode != 0:
                    error_msg = stderr[:500] if stderr else "Unknown FFmpeg error"
                    raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {error_msg}")
                
                if not os.path.exists(output_path):
                    raise RuntimeError("Output file not created")
                
                return (video_id, output_path, "completed", None)
                
            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError("FFmpeg process timed out after 5 minutes")
            
        except Exception as e:
            error_msg = f"ID {video_id} failed: {str(e)}"
            logger.error(error_msg)
            return (video_id, None, f"error: {error_msg}", e)

    def run(self):
        try:
            # Load and validate CSV
            try:
                df = pd.read_csv(self.config['CSV_PATH'])
                if 'ID' not in df.columns:
                    raise ValueError("CSV missing required 'ID' column")
            except Exception as e:
                raise ValueError(f"Error loading CSV: {str(e)}")
            
            # Initialize status columns if missing
            if 'VideoStatus' not in df.columns:
                df['VideoStatus'] = ''
            if 'VideoPath' not in df.columns:
                df['VideoPath'] = ''
            
            pending_rows = df[df['VideoStatus'].str.lower() != 'completed']
            total_tasks = len(pending_rows)
            
            if total_tasks == 0:
                logger.info("No videos to process - all marked as completed")
                return True
                
            logger.info(f"Starting processing of {total_tasks} videos")
            
            with tqdm(total=total_tasks, desc="🚀 Generating Videos", unit="vid") as pbar:
                with ThreadPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
                    futures = {
                        executor.submit(self.process_video, row['ID']): row['ID']
                        for _, row in pending_rows.iterrows()
                    }
                    
                    for future in as_completed(futures):
                        video_id, output_path, status, error = future.result()
                        idx = df[df['ID'] == video_id].index[0]
                        df.at[idx, 'VideoStatus'] = status
                        if output_path:
                            df.at[idx, 'VideoPath'] = output_path
                        pbar.update(1)
                        pbar.set_description(f"Processing ID {video_id}")
            
            # Save results
            try:
                df.to_csv(self.config['CSV_PATH'], index=False)
            except Exception as e:
                logger.error(f"Error saving CSV: {str(e)}")
            
            # Print summary
            completed = len(df[df['VideoStatus'] == 'completed'])
            failed = len(df[df['VideoStatus'].str.startswith('error')])
            
            logger.info("\n🎉 Final Results:")
            logger.info(f"✅ Successfully processed: {completed}")
            logger.info(f"❌ Failed: {failed}")
            
            if failed > 0:
                logger.info("\nFailed IDs:")
                for _, row in df[df['VideoStatus'].str.startswith('error')].iterrows():
                    logger.info(f"ID {row['ID']}: {row['VideoStatus']}")
            
            return completed == total_tasks
            
        except Exception as e:
            logger.error(f"Fatal error in run(): {str(e)}")
            return False

if __name__ == "__main__":
    try:
        logger.info("Starting video generation process")
        generator = VideoGenerator()
        success = generator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        sys.exit(1)