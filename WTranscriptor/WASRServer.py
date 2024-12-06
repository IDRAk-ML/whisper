import json
import os
import sys
import asyncio
import pathlib
import websockets
import concurrent.futures
import logging
import numpy as np
from WTranscriptor.WASR import ASR
from utils.utils import *
import random 
import string

config = {
    "sample_rate": 16000,
    "duration_threshold": 3,
    "vad_threshold": 0.6,
    "model_path": "openai/whisper-base.en",
    'mac_device': True,
}

def initialize_asr_pool(size, config):
    pool = []
    for _ in range(size):
        pool.append(ASR(config))
    return pool

import wave

asr_pool = None
asr_pool_index = 0


def process_chunk(message,asr):
    if isinstance(message,bytes):
        # print(message)
        audio_np = np.frombuffer(message, dtype=np.int16)
        transcript = asr.get_transcript(audio_np)
        # result = delete_file_if_exists(file_name_full)
        print(transcript)
        return transcript[1], True
    else:
        print('Some other instanc')
        return 'Message', True

    
async def recoginize(websocket, path):
    loop = asyncio.get_running_loop()
    global pool
    global asr
    global asr_pool
    global asr_pool_index

    asr_instance = asr_pool[asr_pool_index]
    asr_pool_index = (asr_pool_index + 1) % len(asr_pool)

    logging.info('Connection from %s', websocket.remote_address);
    while True:
        message = await websocket.recv()
        # print(message)
        response, stop = await loop.run_in_executor(pool, process_chunk,message,asr_instance)
        await websocket.send(response)
        if stop: break

async def start():

    global model
    global spk_model
    global args
    global pool
    global asr_pool
    # Enable loging if needed
    #
    # logger = logging.getLogger('websockets')
    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO)

    args = type('', (), {})()

    args.interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
    args.port = int(os.environ.get('VOSK_SERVER_PORT', 2702))
    args.model_path = os.environ.get('VOSK_MODEL_PATH', 'model')
    args.spk_model_path = os.environ.get('VOSK_SPK_MODEL_PATH')
    args.sample_rate = float(os.environ.get('VOSK_SAMPLE_RATE', 16000))
    args.max_alternatives = int(os.environ.get('VOSK_ALTERNATIVES', 0))
    args.show_words = bool(os.environ.get('VOSK_SHOW_WORDS', True))

    

    # Gpu part, uncomment if vosk-api has gpu support
    #
    # from vosk import GpuInit, GpuInstantiate
    # GpuInit()
    # def thread_init():
    #     GpuInstantiate()
    # pool = concurrent.futures.ThreadPoolExecutor(initializer=thread_init)

    pool_size = 12  # Example: Set based on your GPU capabilities
    asr_pool = initialize_asr_pool(pool_size, config)
    
    pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
    print(f' Total pool for parallel running {pool}')
    async with websockets.serve(recoginize, args.interface, args.port,max_size=2**20):
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(start())