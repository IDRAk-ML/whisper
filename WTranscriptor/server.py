# Standard Library Imports
from typing import List
import json
import timeit
from classification_utils.path_config import *
import sys
sys.path.append(CLASSIFIER_MODULE_PATH)
# External Libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException,File
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from WTranscriptor.WASR import ASR
from pydantic import BaseModel
import numpy as np
from utils.utils import *
import random
import string
import librosa
import soundfile as sf
import requests
class AudioInput(BaseModel):
    audio_bytes_str: str


import gzip
from classification_utils.utils import *
# Initialize FastAPI app
app = FastAPI()


def compress_data(data):
    return gzip.compress(data)
suppress_low = [
    "ike and ",
    "lease sub",
    "The end.",
    "ubscribe",
    "my channel",
    "the channel",
    "our channel",
    "ollow me on",
    "for watching",
    "hank you for watching",
    "for your viewing",
    "r viewing",
    "Amara",
    "next video",
    "full video",
    "ranslation by",
    "ranslated by",
    "ee you next week",
    "video",
    'hhhh',
    'bird',
    


    
]
from config import config
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


async def transcript_generator(wave,sampling_rate=16000):
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


@app.websocket("/transcribe")
async def websocket_endpoint_transcription(websocket: WebSocket):
    await manager.connect(websocket)
    audio_buffer = bytearray()  # Use bytearray to accumulate binary data

    
    while True:
        # print(len(audio_buffer))
        data = await websocket.receive()
        # print(data)
        # Check for the type of message received
        if 'text' in data:
            text_data = data['text']
            # print(text_data)
            # If the completion signal is received, convert buffer to numpy array
            if text_data == "E":
                # print(len(audio_buffer))
                numpy_array = np.frombuffer(audio_buffer, dtype=np.int16)  # Assuming float32 dtype for audio
                # print(numpy_array)
                # Clear the audio buffer for the next stream
                audio_buffer = bytearray()
                hal_flag = False
                entity = None
                # Get transcript
                temp_transcript = [[], '']
                classification_result = ''
                transcript = await transcript_generator(wave=numpy_array)
                for hal in suppress_low:
                    if hal in transcript[1]:
                        hal_flag = True
                if hal_flag:
                    transcript = temp_transcript
                    hal_flag = False
                elif len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
                    transcript = temp_transcript
                elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
                    transcript = temp_transcript
                else:
                    transcript = transcript
                    # classification_result = get_classification(transcript[1])
                    # entity = get_entity(transcript[1])
                
                # Send back a structured response
                response_data = {
                    "status": "success",
                    "transcript": {
                        'transcript': transcript[1],
                    }
                }
                compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                await manager.send_data(compressed_data, websocket)

            elif text_data == "P":
                # Handle other text data if needed
                response_data = {
                    "status": "P",
                }
                compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                await manager.send_data(compressed_data, websocket)
            
            else:
                break
                

        elif 'bytes' in data:
            # Append binary data to audio buffer
            audio_buffer.extend(data['bytes'])
            # print(audio_buffer)

        else:
            break
            


    # except WebSocketDisconnect:
    #     manager.disconnect(websocket)
    #     print("WebSocket disconnected")
        
    # except Exception as e:
    #     print(f"Error occurred: {e}")
        
@app.post("/asr_classify_ner")
async def classify_asr(request: Request):
    t1 = timeit.timeit()
    # Receive JSON data
    data = await request.json()
    # print(data)
    # Check if 'audio_data' key is in the received data
    if "audio_data" not in data:
        raise HTTPException(status_code=400, detail="audio_data key not found in request body")

    # Convert received data to numpy array
    numpy_array = np.array(json.loads(data["audio_data"]))

    # Get transcript
    temp_transcript = [[], '']
    classification_result = ''
    transcript = await transcript_generator(wave=numpy_array)
    # print(transcript)
    if len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
        transcript = temp_transcript
    elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
        transcript = temp_transcript
    else:
        transcript = transcript
    t2 = timeit.timeit()
    print('Time taken', t2-t1)
    # Return a structured response
    response_data = {
        "status": "success",
        "transcript": {'transcript':transcript[1]}
    }
    return JSONResponse(content=response_data)

@app.websocket("/asr_classify_chunks")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    audio_buffer = bytearray()  # Use bytearray to accumulate binary data

    try:
        while True:
            # print(len(audio_buffer))
            data = await websocket.receive()
            # print(data)
            # Check for the type of message received
            if 'text' in data:
                text_data = data['text']
                # print(text_data)
                # If the completion signal is received, convert buffer to numpy array
                if text_data == "E":
                    # print(len(audio_buffer))
                    numpy_array = np.frombuffer(audio_buffer, dtype=np.int16)  # Assuming float32 dtype for audio
                    # print(numpy_array)
                    # Clear the audio buffer for the next stream
                    audio_buffer = bytearray()
                    hal_flag = False
                    entity = None
                    # Get transcript
                    temp_transcript = [[], '']
                    classification_result = ''
                    transcript = await transcript_generator(wave=numpy_array)
                    for hal in suppress_low:
                        if hal in transcript[1]:
                            hal_flag = True
                    if hal_flag:
                        transcript = temp_transcript
                        hal_flag = False
                    elif len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
                        transcript = temp_transcript
                    elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
                        transcript = temp_transcript
                    else:
                        transcript = transcript
                        classification_result = get_classification(transcript[1])
                        entity = get_entity(transcript[1])

                    # Send back a structured response
                    response_data = {
                        "status": "success",
                        "transcript": {
                            'transcript': transcript[1],
                            'intent': classification_result,
                            'entity': entity
                        }
                    }
                    compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                    await manager.send_data(compressed_data, websocket)

                elif text_data == "P":
                    # Handle other text data if needed
                    response_data = {
                        "status": "P",
                    }
                    compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                    await manager.send_data(compressed_data, websocket)
                
                else:
                    break
                    

            elif 'bytes' in data:
                # Append binary data to audio buffer
                audio_buffer.extend(data['bytes'])
                # print(audio_buffer)

            else:
                break
            


    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")
        
    except Exception as e:
        print(f"Error occurred: {e}")

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
from websockets.exceptions import ConnectionClosedOK


@app.websocket("/ws_file_transcribe1")
async def websocket_endpoint(websocket: WebSocket):
    try:
        model_name = config.get('model_name','whisper')
        await websocket.accept()
        data = await websocket.receive_bytes()  # Receive file data as bytes
        file_name_short = ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + ".wav"
        file_name_full = f'temp/{file_name_short}'
        with open(file_name_full, "wb") as file:
            file.write(data)  # Save the received data to a file

        audio_np,sr = read_wav_as_int16(file_name_full)
        transcript = await transcript_generator(wave=audio_np)
        filtered_transcript = filter_hal(transcript[1])
        await websocket.send_text(f"{filtered_transcript}")
        await websocket.close()
        try:
            result = delete_file_if_exists(file_name_full)
        except:
            pass
    except Exception as e:
        print(f'Error: {e}')
        
@app.websocket("/ws_persistent_transcribe")
async def websocket_persistent_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection.
    try:
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
