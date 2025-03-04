import requests
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample


SERVER_ADDRESS = "148.251.178.29:9012" 

def transcribe_audio(file_path: str):
    # Read the audio file
    data, samplerate = sf.read(file_path)
    
    # Resample if the sampling rate is not 16000 Hz
    if samplerate != 16000:
        num_samples = int(len(data) * 16000 / samplerate)
        data = resample(data, num_samples)
        samplerate = 16000
    
    # Save the resampled audio to a BytesIO buffer
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format='WAV')
    buffer.seek(0)
    
    # Define the API URL
    url = f"http://{SERVER_ADDRESS}/transcribe_array"
    
    # Send the request
    files = {"file": ("audio.wav", buffer, "audio/x-wav")}
    headers = {"accept": "application/json"}
    response = requests.post(url, files=files, headers=headers)
    
    # Return the transcript
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to transcribe audio", "status_code": response.status_code, "response": response.text}

# Example usage:
transcript = transcribe_audio("20sec_DeepFilterNet3.wav")
print(transcript)


'''
example

python test_client.py
{'message': 'Conversion successful', 
'transcript': " We are not asking to make any decisions over the phone.",
 'am_result': ''}

'''