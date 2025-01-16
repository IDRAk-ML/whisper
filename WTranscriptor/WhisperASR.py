import torch
import numpy as np
# from whisper2 import WhisperTranscriptorAPI 
from WTranscriptor.Singlton import SingletonMeta
# from silero1 import SileroTranscriptorAPI
import warnings
warnings.filterwarnings('ignore')


#ASR class defination
class ASR(object):
    """
    
    Whisper Model. Takes numpy wav and return transcript
    Description of functions:

    1. __init__ : Constructor: requires a config dict but also can set default values if not found in config
    2. get_transcript: Takes in a torch array and returns the transcript
    
    """
    def __init__(self, config):
        '''
        ;Loading the vad and whisper model
        
        args:
        ;config (dictionary) having configuration for model loadings
        '''
        # configuring configs
        print(config)
        self.samplerate = config.get("samplerate", 16000)
        self.cuda_device = config.get("cuda_device", "cpu")
        self.check_interval = config.get("check_interval",3)
        #path of whisper-tiny.en; can be whisper-base.en openai/whisper-base.en
        self.model_path= config.get("model_path","tiny.en") 
        self.mac_device = config.get('mac_device',False)
        print("[INFO] Loading Models")
        model_name = config.get('model_name','whisper')
        vad_thresold = config.get('vad_thresold',0.6)
        if model_name == 'whisper':
            from WTranscriptor.whisperlatest import WhisperTranscriptorAPI 
            self.model = WhisperTranscriptorAPI(model_path=self.model_path,mac_device=False,vad_thresold=vad_thresold)
        else:
            from WTranscriptor.nemo_asr import NemoTranscriptorAPI
            self.model = NemoTranscriptorAPI(model_path=self.model_path,mac_device=False)
        # self.model = SileroTranscriptorAPI()
        print("[INFO] Model Loaded")
    @staticmethod
    def get_instance(config):
        return ASR(config)
    async def get_transcript(self, data_torch, is_greedy=False, emissions_only=False,sample_rate=16000,
                             enable_vad = False):
        '''
         This function will generate transcripts using whisper.
         
        '''
        wave=data_torch # get the wave data
        transcript,ids = await self.model.generate_transcript_numpy(wave=wave,sample_rate=sample_rate,enable_vad=enable_vad)
        #ids are model generated tokens id for the ASR
        return ids,transcript
    
    async def transcribe_file(self,audio_data,language):
        resp = await self.model.transcribe_file(audio_data,language)
        #ids are model generated tokens id for the ASR
        return resp

if __name__ == "__main__":
    config = dict()
    config["cuda_device"] = "cpu"
    asr=ASR(config=config)
    