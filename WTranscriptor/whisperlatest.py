from transformers import pipeline,WhisperProcessor, WhisperForConditionalGeneration,WhisperTokenizer
import torch
import timeit
import numpy as np
import warnings
import os
import zlib
from typing import Optional,Dict,List,Collection
warnings.filterwarnings('ignore')
from utils.utils import *
'''
Faster Implementation of Whisper
'''

# torch.set_num_threads(8)


class WhisperTranscriptorAPI:
    '''
    This Module is based on CTC fast Whisper for Audio Transcription.
    We need WhisperProcessor and WhisperConditionalGeneration for 
    CTC task i.e. ASR. 
    example:
          whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-tiny.en')
          
    '''
    #----------------------- constructor -------------------------------------
    #
    

    def __init__(self,model_path='',file_processing=False,word_timestamp=True,mac_device=False,
                 dtype = torch.float16,en_flash_attention = False,batch_size=128,
                 vad_model = None,vad_thresold = 0.6,detect_language = True):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''
        self.mac_device = mac_device
        self.model_path = model_path
        self.vad_thresold = vad_thresold
        self.batch_size = batch_size
        self.detect_language = detect_language
        # print(self.model_path)
        # device='cpu'
        # compute_type="int8"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(device == "cuda" , "cuda check")
        if mac_device:
            print(f"[INFO] Loading on Mac Device")
            self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device="mps" if mac_device else f"cuda:{cuda_device_id}",
                )
        else:
            if device == 'cuda':
                cuda_device_id = 0
                print(f"[INFO] Loading {self.model_path} on Cuda")
                try:
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device="mps" if mac_device else f"cuda:{cuda_device_id}",
                    model_kwargs={"attn_implementation": "flash_attention_2"} if en_flash_attention else {"attn_implementation": "sdpa"},
                        )
                except ValueError:
                    print("[INFO] Cuda Support Issue Moving to CPU")
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device=device)
            else:
                    print("[INFO] Loading on CPU")
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device=device)
        self.OUTPUT_DIR= "audios"
        self.vad_model, self.utils = torch.hub.load('snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
        self.vad_model = self.vad_model.to(device)
        (self.get_speech_timestamps,
        self.save_audio,
        self.read_audio,
        self.VADIterator,
        self.collect_chunks) = self.utils

        
        
    
    #-------------------- generate transcript from nmpy array ----------------
    #

    async def generate_transcript_numpy(self, wave,sample_rate=16000,enable_vad = False):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''
        

        if self.mac_device:
            torch.mps.empty_cache()
        generate_kwargs = {"task": 'transcribe', "language": 'en'}
        if self.model_path.split(".")[-1] == "en":
            generate_kwargs.pop("task")
            generate_kwargs.pop("language") 
         
        t1 = timeit.default_timer()
        if enable_vad:
            wave = torch.from_numpy(wave).to(device=self.device).float()
            speech_timestamps = self.get_speech_timestamps(wave, self.vad_model, sampling_rate=16000,threshold=self.vad_thresold)
        else:
            speech_timestamps = True
        print('vad',enable_vad,speech_timestamps)
        if speech_timestamps:
            if enable_vad:
                wave1 = self.collect_chunks(speech_timestamps, wave)
                # print(wave1)
                wave = wave1.cpu().numpy()
            else:
                pass
            t1 = timeit.default_timer()
            outputs = self.model(
                wave,
            chunk_length_s=15,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
                 )
            transcription = filter_hallucination(outputs['text'])
            print(transcription)
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]
    async def genereate_transcript_from_file(self, file_name):
        return 'method not implemented; for whisper use generate_transcript_numpy', []
