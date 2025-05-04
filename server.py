from typing import List, Optional
import json
import gzip
import random
import string
import os
import numpy as np
import librosa
import soundfile as sf
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from WTranscriptor.utils.utils import *
from WTranscriptor.classification_utils.utils import *
from hallucination_filters import suppress_low
from config import config, HELPING_ASR_FLAG,SMART_AM_CHECK,ENV_DOCKER
import io
# Initialize FastAPI app
app = FastAPI()


helping_asr = None
if HELPING_ASR_FLAG:
    from WTranscriptor.SVClient import ASRClient
    helping_asr = ASRClient()

class AudioInput(BaseModel):
    audio_bytes_str: str

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    processing_time: float

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_data(self, data: str, websocket: WebSocket):
        await websocket.send_text(data)

manager = ConnectionManager()

def compress_data(data):
    return gzip.compress(data)

def filter_hal(txt: str) -> str:
    hal = ['you', 'your', 'video', 'thank','bye']
    return '' if len(txt) < 6 and any(word in txt for word in hal) else txt




def check_am(file_audio: bytes) -> bool:
    if SMART_AM_CHECK: 
        if ENV_DOCKER:
            url = "http://backend8034:8034/detect-smart-am/"
        else:
            url = "http://127.0.0.1:8034/detect-smart-am/" 
        temp_file = save_byte_to_temp_file(file_audio=file_audio)
        if tempfile:
            files = {"file": open(temp_file, "rb")}
            response = requests.post(url, files=files)
            
            response = response.json()
            print('AM Response',response)
            return response.get("match_detected",False) 
    
    return False  # Placeholder for external request logic

@app.post("/transcribe_array1")
async def audio_to_numpy(file: bytes = File(...)):
    try:
        # Step 1: Load as int16
        audio_np, sampling_rate = sf.read(io.BytesIO(file), dtype='int16')

        if sampling_rate != 16000:
            print(f"[+] Original SR: {sampling_rate}, resampling to 16000 Hz")

            # Step 2: Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32)
            audio_float /= np.max(np.abs(audio_float)) + 1e-6  # avoid division by zero

            # Step 3: Resample
            audio_resampled = librosa.resample(audio_float, orig_sr=sampling_rate, target_sr=16000)

            # Step 4: Convert back to int16
            audio_np = (audio_resampled * 32767).astype(np.int16)
            sampling_rate = 16000

        print(f"[+] Final Audio Shape: {audio_np.shape}, Dtype: {audio_np.dtype}")

        # Now safe to proceed with your pipeline
        am_result = check_am(file)

        transcript = await transcript_generator(wave=audio_np, sampling_rate=16000)
        txt = filter_hal(transcript[1])

        if len(txt) <= 1 and helping_asr:
            txt = await helping_asr.transcribe_audio_array(audio_array=audio_np)

        print(f'[+] Transcript Sending {txt if len(txt) > 3 else "Nothing"}')
        return {"message": "Conversion successful", "transcript": txt, "am_result": am_result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/transcribe_array")
async def audio_to_numpy(file: bytes = File(...)):
    try:
        am_result = check_am(file)
        audio_np = np.frombuffer(file, dtype=np.int16)
        print('Wave type init',audio_np)
        transcript = await transcript_generator(wave=audio_np, sampling_rate=16000)
        txt = filter_hal(transcript[1])

        if len(txt)<=1 and helping_asr:
            txt = await helping_asr.transcribe_audio_array(audio_array=audio_np)

        
        print(f'[+] Transcript Sending {txt if len(txt) > 3 else "Nothing"}')
        return {"message": "Conversion successful", "transcript": txt, "am_result": am_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...), language: Optional[str] = "en"):
    try:
        contents = await audio_file.read()
        audio_array, sampling_rate = sf.read(contents, dtype='int16')
        return TranscriptionResponse(text="response", language=language, duration=0.0, processing_time=0.0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws_file_transcribe2")
async def websocket_endpoint(websocket: WebSocket):
    # try:
        await websocket.accept()
        data = await websocket.receive_bytes()
        file_name = f"temp/{''.join(random.choices(string.ascii_letters + string.digits, k=6))}.wav"
        os.makedirs('temp', exist_ok=True)
        with open(file_name, "wb") as file:
            file.write(data)
        
        
        audio_np, sr = read_wav_as_int16(file_name)
        print('Wave type init',audio_np)
        print(sr)
        transcript = await transcript_generator(wave=audio_np,sampling_rate=sr)
        filtered_transcript = filter_hal(transcript[1])

        if len(filtered_transcript)<=1 and helping_asr:
            filtered_transcript = await helping_asr.transcribe_audio_array(audio_array=audio_np)
        print(f'[+] Transcript Sending {filtered_transcript if len(filtered_transcript) > 3 else "Nothing"}')
        await websocket.send_text(filtered_transcript)
    # except Exception as e:
    #     print(f"Error: {e}")
    # finally:
    #     await websocket.close()
    #     try:
    #         os.remove(file_name)
    #     except FileNotFoundError:
    #         pass



@app.websocket("/ws_file_transcribe1")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        data = await websocket.receive_bytes()  # Receive audio data as bytes
        
        # Convert received bytes directly to a NumPy array (like REST API)
        audio_np = np.frombuffer(data, dtype=np.int16)
        
        print('Wave type init', audio_np)

        transcript = await transcript_generator(wave=audio_np, sampling_rate=16000)  # Ensure same sample rate
        filtered_transcript = filter_hal(transcript[1])

        if len(filtered_transcript) <= 1 and helping_asr:
            filtered_transcript = helping_asr.transcribe_audio_array(audio_array=audio_np)

        print(f'[+] Transcript Sending {filtered_transcript if len(filtered_transcript) > 3 else "Nothing"}')
        await websocket.send_text(filtered_transcript)
    except Exception as E:
        print('Error',E)
    finally:
        await websocket.close()
        try:
            os.remove('')
        except FileNotFoundError:
            pass 

    
@app.websocket("/ws_persistent_transcribe")
async def websocket_persistent_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                file_data = data["bytes"]
                file_name = f"temp/{''.join(random.choices(string.ascii_letters + string.digits, k=6))}.wav"
                with open(file_name, "wb") as file:
                    file.write(file_data)
                
                audio_np, sr = read_wav_as_int16(file_name)
                transcript = await transcript_generator(wave=audio_np)
                filtered_transcript = filter_hal(transcript[1])
                
                if len(filtered_transcript)<=1 and helping_asr:
                    filtered_transcript = await helping_asr.transcribe_audio_array(audio_array=audio_np)

                
                print(f'[+] Transcript Sending {filtered_transcript if len(filtered_transcript) > 3 else "Nothing"}') 
                await websocket.send_text(filtered_transcript)
                os.remove(file_name)
            elif "text" in data:
                await websocket.send_text(f"Received text message: {data['text']}")
            else:
                await websocket.send_text("Unsupported message type.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
