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
from WTranscriptor.WhisperASR import ASR
from WTranscriptor.utils.utils import *
from WTranscriptor.classification_utils.utils import *
from hallucination_filters import suppress_low
from config import config, HELPING_ASR_FLAG

# Initialize FastAPI app
app = FastAPI()

# Initialize ASR model
asr = ASR.get_instance(config)
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

def check_am(file_audio: bytes) -> str:
    return ''  # Placeholder for external request logic

async def transcript_generator(wave: np.ndarray, sampling_rate: int = 16000, file_mode: bool = False, language: str = 'en'):
    if not file_mode:
        model_name = config.get('model_name', 'whisper')
        wave = wave / np.iinfo(np.int16).max
        if sampling_rate != 16000:
            wave = librosa.resample(wave, orig_sr=sampling_rate, target_sr=16000)
        transcript = await asr.get_transcript(wave, sample_rate=sampling_rate, enable_vad=config['enable_vad'])
    else:
        transcript = await asr.transcribe_file(wave, language=language)
    return transcript

@app.post("/transcribe_array")
async def audio_to_numpy(file: bytes = File(...)):
    try:
        am_result = check_am(file)
        audio_np = np.frombuffer(file, dtype=np.int16)
        transcript = await transcript_generator(wave=audio_np, sampling_rate=16000)
        txt = filter_hal(transcript[1])

        if len(txt)<=1 and helping_asr:
            txt = helping_asr.transcribe_audio_array(audio_array=audio_np)
        
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

@app.websocket("/ws_file_transcribe1")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        data = await websocket.receive_bytes()
        file_name = f"temp/{''.join(random.choices(string.ascii_letters + string.digits, k=6))}.wav"
        os.makedirs('temp', exist_ok=True)
        with open(file_name, "wb") as file:
            file.write(data)
        
        audio_np, sr = read_wav_as_int16(file_name)
        transcript = await transcript_generator(wave=audio_np)
        filtered_transcript = filter_hal(transcript[1])

        if len(filtered_transcript)<=1 and helping_asr:
            filtered_transcript = helping_asr.transcribe_audio_array(audio_array=audio_np)

        await websocket.send_text(filtered_transcript)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
        try:
            os.remove(file_name)
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
                    filtered_transcript = helping_asr.transcribe_audio_array(audio_array=audio_np)
                
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
