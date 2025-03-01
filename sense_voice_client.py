import requests

API_URL = "http://78.46.99.15:9005/api/v1/asr"

def send_audio_to_asr(audio_files, keys, lang="auto"):
    """
    Sends audio files to ASR API for transcription.

    Parameters:
    - audio_files: List of file paths (wav or mp3, 16KHz)
    - keys: List of names for each audio file
    - lang: Language of audio content (default: "auto")

    Returns:
    - JSON response from ASR API
    """
    files = [("files", (file, open(file, "rb"), "audio/wav")) for file in audio_files]
    data = {
        "keys": ",".join(keys),
        "lang": lang
    }
    
    response = requests.post(API_URL, files=files, data=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    audio_files = ["WTranscriptor/audios/40sec.wav"]  # Replace with your actual file paths
    keys = ["audio1"]
    lang = "en"

    result = send_audio_to_asr(audio_files, keys, lang)
    print(result)


'''
{'result': [{'key': 'audio1', 'text': 'we the people of the united states in order to form a more perfect union establish', 'raw_text': '<|en|><|NEUTRAL|><|Speech|><|woitn|>we the people of the united states in order to form a more perfect union establish', 'clean_text': 'we the people of the united states in order to form a more perfect union establish'}]}
'''