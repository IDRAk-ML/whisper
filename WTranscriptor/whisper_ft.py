import timeit
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from WTranscriptor.utils.utils import *
from pydantic import BaseModel
import time
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment
'''
Faster Implementation of Whisper
'''
import tempfile
import random
import string


def generate_random_filename(extension="mp3", length=10):
    random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{random_name}.{extension}"

def save_wave_as_mp3(wave, frame_rate=16000):
    """
    Save the wave numpy array as an MP3 file with a random name in a temporary directory.
    
    Args:
        wave (numpy.ndarray): The wave data to save.
        frame_rate (int): The frame rate of the audio.
        
    Returns:
        str: The path to the saved MP3 file.
    """
    temp_dir = tempfile.gettempdir()
    random_name = generate_random_filename()
    temp_path = os.path.join(temp_dir, random_name)

    # Convert numpy array to AudioSegment
    audio_segment = AudioSegment(
        wave.tobytes(), 
        frame_rate=frame_rate,
        sample_width=wave.dtype.itemsize, 
        channels=1
    )

    # Export as mp3
    audio_segment.export(temp_path, format="mp3")
    print(f"Saved MP3 to {temp_path}")
    
    return temp_path


# torch.set_num_threads(8)


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    processing_time: float




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
                 en_flash_attention = False,batch_size=128,
                 vad_model = None,vad_thresold = 0.4,detect_language = True):

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
        
        device = 'cuda'
        self.device = device
        print(device == "cuda" , "cuda check")
        if mac_device:
            print(f"[INFO] Loading on Mac Device")
            self.model = WhisperModel(self.model_path, device="cuda", compute_type="float16")
        else:
            if device == 'cuda':
                cuda_device_id = 0
                print(f"[INFO] Loading {self.model_path} on Cuda")
                try:
                    self.model = WhisperModel(self.model_path, device=self.device, compute_type="float16")
                except ValueError:
                    print("[INFO] Cuda Support Issue Moving to CPU")
                    self.model = WhisperModel(self.model_path, device="cpu", compute_type="float16")
            else:
                    self.model = WhisperModel(self.model_path, device="cpu", compute_type="float16") 
        self.OUTPUT_DIR= "audios"
        
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
        

        
        
         
        t1 = timeit.default_timer()
        speech_timestamps = True 
        if speech_timestamps:

            mp3_path = save_wave_as_mp3(wave)
            t1 = timeit.default_timer()
            print(mp3_path)
            segments, info = self.model.transcribe(
                                mp3_path,
                                beam_size=5,
                            language="ur",
                                )
            text = ""
            for segment in segments:
                text += segment.text

            transcription = filter_hallucination(text)
            
            print(transcription)
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]
    async def genereate_transcript_from_file(self, file_name):
        return 'method not implemented; for whisper use generate_transcript_numpy', []


    async def transcribe(self, audio_data, language: str = "en") -> TranscriptionResponse:
        start_time = time.time()
        
        # Create an in-memory buffer for the audio data
        
        
        # Transcribe the audio
        outputs = self.model.transcribe(
            audio_data,
            beam_size=5,
        )

        # Combine all segments
        text = outputs['text']
        
        processing_time = time.time() - start_time
        

        return TranscriptionResponse(
            text=text,
            language=language,
            duration=1.2,
            processing_time=processing_time
        )

        