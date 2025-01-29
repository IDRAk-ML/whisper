# Standard Library Imports
from typing import List
import json
import timeit
from WTranscriptor.classification_utils.path_config import *
import sys
sys.path.append(CLASSIFIER_MODULE_PATH)
# External Libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException,File
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from WTranscriptor.WhisperASR import ASR
from pydantic import BaseModel
import numpy as np
from WTranscriptor.utils.utils import *
import random
import string
import librosa
import soundfile as sf
import requests
from config import config
import gzip
from WTranscriptor.classification_utils.utils import *
from fastapi import FastAPI, UploadFile, File
from hallucination_filters import suppress_low

from typing import Optional

class AudioInput(BaseModel):
    audio_bytes_str: str



# Initialize FastAPI app
app = FastAPI()


def compress_data(data):
    return gzip.compress(data)


asr = ASR.get_instance(config)



class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts and stores a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection."""
        self.active_connections.remove(websocket)

    async def send_data(self, data: str, websocket: WebSocket):
        """Sends data through a given WebSocket connection."""
        await websocket.send_text(data)




    
manager = ConnectionManager()


async def transcript_generator(wave,sampling_rate=16000,file_mode=False,language='en'):

    if not file_mode:
        model_name = config.get('model_name','whisper')
        wave = wave / np.iinfo(np.int16).max
        if sampling_rate != 16000:
            wave = librosa.resample(wave, orig_sr=sampling_rate, target_sr=16000)



        transcript = [[],'']
        if model_name == 'whisper':
            transcript = await asr.get_transcript(wave,sample_rate=sampling_rate,enable_vad=config['enable_vad'])
        else:
            file_name = save_wav_sync(wave)
            transcript = await asr.get_transcript_from_file(file_name=file_name)
        return transcript
    else:
        transcript = await asr.transcribe_file(wave,language=language)
        return transcript



def filter_hal(txt):
    hal = ['you','your','video','thank']
    if len(txt) < 6:
        for hal_st in hal:
            if hal_st in txt:
                return ''
    return txt 

def check_am(file_audio):
    # '''
    # send the file data to server
    # '''
    # files = {'file': ('audio.raw', file_audio, 'application/octet-stream')}
    # url = 'http://127.0.0.1:3334/compare_audio'
    # response = requests.post(url, files=files)
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return response.text
    return ''
    
@app.post("/transcribe_array")
async def audio_to_numpy(file: bytes = File(...)):
    # try:
        am_result = check_am(file)
        audio_np = np.frombuffer(file, dtype=np.int16)
        transcript = await transcript_generator(wave=audio_np,sampling_rate=16000)
        txt = filter_hal(transcript[1])
        return {"message": "Conversion successful", "transcript":txt,'am_result':am_result}
    # except Exception as e:
        # raise HTTPException(status_code=400, detail=str(e))


        


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    processing_time: float

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: Optional[str] = "en"
):
    """
    Endpoint to transcribe audio files.
    Accepts WAV files and returns transcription results.
    """
    print('Transcribed Call') 
    contents = await audio_file.read()
    audio_array, sampling_rate = sf.read(contents, dtype='int16') 
    
    # response = await asr.transcribe_file(audio_array,language=language)
    print('response')
    return 'response'

