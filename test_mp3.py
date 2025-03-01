import requests
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample
from pydub import AudioSegment

SERVER_ADDRESS = "5.9.50.177:9004"

def convert_mp3_to_wav(mp3_path: str):
    """Convert an MP3 file to WAV format and return the byte buffer."""
    audio = AudioSegment.from_mp3(mp3_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def transcribe_audio(file_path: str):
    """Transcribe an audio file (MP3 or WAV)."""
    
    # Check file format
    if file_path.lower().endswith(".mp3"):
        audio_buffer = convert_mp3_to_wav(file_path)
    elif file_path.lower().endswith(".wav"):
        audio_buffer = open(file_path, "rb")
    else:
        return {"error": "Unsupported file format. Only MP3 and WAV are allowed."}

    # Read audio and resample if needed
    data, samplerate = sf.read(audio_buffer)
    if samplerate != 16000:
        num_samples = int(len(data) * 16000 / samplerate)
        data = resample(data, num_samples)
        samplerate = 16000

    # Convert to WAV format in memory
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format='WAV')
    buffer.seek(0)

    # Define the API URL
    url = f"http://{SERVER_ADDRESS}/transcribe_array"

    # Send request
    files = {"file": ("audio.wav", buffer, "audio/x-wav")}
    headers = {"accept": "application/json"}
    response = requests.post(url, files=files, headers=headers)

    # Return response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to transcribe audio", "status_code": response.status_code, "response": response.text}


# Example usage:
transcript = transcribe_audio("WTranscriptor/audios/output.mp3")
print(transcript)
