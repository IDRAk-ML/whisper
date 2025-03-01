import subprocess

def convert_wav_to_mp3_ffmpeg(wav_path: str, mp3_path: str, bitrate="192k"):
    """Convert WAV to MP3 using FFmpeg."""
    command = [
        "ffmpeg",
        "-i", wav_path,       # Input WAV file
        "-b:a", bitrate,      # Bitrate (default: 192k)
        "-y",                 # Overwrite existing file
        mp3_path              # Output MP3 file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful: {mp3_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

# Example usage
convert_wav_to_mp3_ffmpeg("WTranscriptor/audios/20sec.wav", "WTranscriptor/audios/20sec.wav")
