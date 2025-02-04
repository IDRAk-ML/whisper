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
    try:
        am_result = check_am(file)
        audio_np = np.frombuffer(file, dtype=np.int16)
        transcript = await transcript_generator(wave=audio_np,sampling_rate=16000)
        txt = filter_hal(transcript[1])
        return {"message": "Conversion successful", "transcript":txt,'am_result':am_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


        


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

@app.websocket("/ws_file_transcribe1")
async def websocket_endpoint(websocket: WebSocket):
    try:
        print('transcript 1 called')
        model_name = config.get('model_name','whisper')
        await websocket.accept()
        data = await websocket.receive_bytes()  # Receive file data as bytes
        file_name_short = ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + ".wav"
        os.makedirs('temp',exist_ok=True)
        file_name_full = f'temp/{file_name_short}'
        with open(file_name_full, "wb") as file:
            file.write(data)  # Save the received data to a file

        audio_np,sr = read_wav_as_int16(file_name_full)
        transcript = await transcript_generator(wave=audio_np)
        filtered_transcript = filter_hal(transcript[1])
        print('Transcript:',filtered_transcript)
        await websocket.send_text(f"{filtered_transcript}")
        await websocket.close()
        try:
            result = delete_file_if_exists(file_name_full)
            print('Deleted',result)
        except:
            pass
    except Exception as e:
        print(f'Error: {e}')
        
@app.websocket("/ws_persistent_transcribe")
async def websocket_persistent_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection.
    try:
        print('ws_persistent_transcribe')
        while True:  # Keep the connection open until the client closes it.
            data = await websocket.receive()  # Wait for a message from the client.
            
            if "bytes" in data:  # Check if the message is a bytes instance.
                file_data = data["bytes"]
                file_name_short = ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + ".wav"
                file_name_full = f'temp/{file_name_short}'
                with open(file_name_full, "wb") as file:
                    file.write(file_data)  # Save the received data to a file.
                
                # Process the audio file to transcribe it.
                try:
                    audio_np, sr = read_wav_as_int16(file_name_full)
                    transcript = await transcript_generator(wave=audio_np)
                    filtered_transcript = filter_hal(transcript[1])
                    await websocket.send_text(f"{filtered_transcript}")
                finally:
                    try:
                        os.remove(file_name_full)  # Attempt to delete the file after processing.
                    except Exception as e:
                        print(f"Failed to delete file {file_name_full}: {e}")
            else:
                # Handle non-bytes messages here.
                # For this example, we're just echoing back the text.
                if "text" in data:
                    await websocket.send_text(f"Received text message: {data['text']}")
                else:
                    await websocket.send_text("Unsupported message type.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()  # Make sure the WebSocket is closed properly.