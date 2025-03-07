import numpy as np
import wave
import requests
import os
import uuid
import librosa
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file


import torch

class ASRClient:
    def __init__(self, api_url="http://0.0.0.0:9006/api/v1/asr", temp_dir="temp"):
        """
        Initializes the ASRClient with API URL and temp directory.

        Parameters:
        - api_url: URL of the ASR API
        - temp_dir: Directory to store temporary audio files
        """
        self.api_url = api_url
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.denoise_model, self.df_state, _ = init_df()


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vad_model, self.utils = torch.hub.load('snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
        self.vad_model = self.vad_model.to(self.device)
        (self.get_speech_timestamps,
        self.save_audio_,
        self.read_audio_,
        self.VADIterator,
        self.collect_chunks) = self.utils
        


    def filter_hallucination(self,text):
        if text == 'ju' or text =='y' or text =='i':
            return ''
        return text
    

    def resample_audio(self,audio_array,orig_sr=16000,target_sr=16000):

        wave_file = audio_array
        
        if orig_sr != target_sr:
            wave_file = librosa.resample(wave_file, orig_sr=orig_sr, target_sr=target_sr)

        return wave_file
    


    def save_audio(self, audio_array, file_name="audio.wav", sample_rate=16000):
        """
        Saves a NumPy array as a WAV file inside the `temp/` directory.

        Parameters:
        - audio_array: NumPy array containing audio data
        - file_name: Name of the file to save (without directory path)
        - sample_rate: Sampling rate (default: 16000 Hz)

        Returns:
        - Path to the saved WAV file
        """
        
        wave_file = self.resample_audio(audio_array,sample_rate,16000)
        
        file_path = os.path.join(self.temp_dir, file_name)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)
            wf.writeframes(np.int16(wave_file).tobytes())
        return file_path

    def send_audio_to_asr(self, audio_path, key, lang="en"):
        """
        Sends a WAV file to the ASR API for transcription.

        Parameters:
        - audio_path: Path to the WAV file
        - key: Identifier for the audio file
        - lang: Language of audio content (default: "auto")

        Returns:
        - JSON response from ASR API
        """
        with open(audio_path, "rb") as file:
            files = [("files", (audio_path, file, "audio/wav"))]
            data = {
                "keys": key,
                "lang": lang
            }
            response = requests.post(self.api_url, files=files, data=data)
            return response.json()
    def denoise_audio(self,audio_path):
        print('[+] Denoise Audio')
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        enhanced = enhance(self.denoise_model, self.df_state, audio)
        # Save for listening
        save_audio(audio_path, enhanced, self.df_state.sr())
        return audio_path 
    
    
    
    def apply_vad(self,audio_path):
        print('APPLYING SEC VAD')
        wave, sr = librosa.load(audio_path, sr=None)

        if sr != 16000:
            wave = self.resample_audio(wave,sr,16000)

        wave = torch.from_numpy(wave).to(device=self.device).float()
        speech_timestamps = self.get_speech_timestamps(wave, self.vad_model, sampling_rate=16000,threshold=0.3)
        print(speech_timestamps)
        if speech_timestamps:
            wave1 = self.collect_chunks(speech_timestamps, wave)
            # print(wave1)
            wave = wave1.cpu().numpy()
            # self.save_audio(wave,audio_path,16000)
            return audio_path,True
        else:
            print('VAD Did Not Detect a Speech')
            return audio_path,False
        

    def transcribe_audio_array(self, audio_array, sample_rate=16000, lang="en"):
        """
        Converts a NumPy array to a WAV file, sends it to ASR API, and returns only the transcript.

        Parameters:
        - audio_array: NumPy array containing audio data
        - sample_rate: Sampling rate (default: 16000 Hz)
        - lang: Language for ASR (default: "auto")

        Returns:
        - Transcription text if available, otherwise an empty string
        """
        key = str(uuid.uuid4())  # Generate a random UUID as the key
        audio_path = self.save_audio(audio_array, f"{key}.wav", sample_rate)

        # denoise here
        audio_path = self.denoise_audio(audio_path)

        audio_path,_ = self.apply_vad(audio_path)

        if not audio_path:
            return ""
        
        response = self.send_audio_to_asr(audio_path, key, lang)
        print("[-] SV Client is predicting",response)
        # Extract transcript
        if "result" in response and isinstance(response["result"], list):
            for entry in response["result"]:
                if entry.get("key") == key and "clean_text" in entry:
                    
                    if _:
                        return self.filter_hallucination(entry["clean_text"])
                    else:
                        def hal_check(text: str) -> str:
                            keep = ['hel','ye','hi','yes','yup','um','hello','hey','he']
                            for k in keep:
                                if k in text:
                                    if 'the' in text:
                                        return ""
                                    return text
                            
                            text = text.strip()
                            hal_word = ['it','the','ug','-','ok']
                            for h in hal_word:
                                if h in text:
                                    return ""
                            if len(text) >= 2:
                                return text.strip()
                            return ''
                        
                        text = hal_check(entry["clean_text"])
                        
                        return text
                        
        return ""  # Return empty string if transcript is not found


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Simulated audio signal (1 second of silence at 16KHz)
    sample_rate = 16000
    duration = 1  # seconds
    audio_data = np.zeros(sample_rate * duration, dtype=np.int16)

    asr_client = ASRClient()
    transcript = asr_client.transcribe_audio_array(audio_data, sample_rate=sample_rate, lang="en")

    print(f"Transcript: '{transcript}'")  # Prints transcript or empty string if unavailable
