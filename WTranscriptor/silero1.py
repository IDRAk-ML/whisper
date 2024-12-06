from sil_asr.src.utils import *
import warnings
import torch
torch._C._jit_set_bailout_depth(0)
from glob import glob
from omegaconf import OmegaConf
import numpy as np
import onnx
import onnxruntime
warnings.filterwarnings('ignore')


class SileroTranscriptorAPI:
    '''
    Silero Transcriptopr
    '''
    def __init__(self,model_path='',load_onnx=True):

        '''
        An ASR Based with some Modification from Silero
        '''
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Loading on ',self.device)
        models_list = OmegaConf.load('ali_silero_asr/models.yml')    
        self.model, self.decoder = init_jit_model(models_list.stt_models.en.latest.jit, device=self.device)

    def generate_transcript_numpy(self, wave):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''
        transcription = self.__forward__(wave=wave)
        return transcription,[]
            
    def __forward__(self,wave):
        '''
        Forward Method to do everything internally
        '''
        processed_wave = process_audio(wave,original_sr=16000,target_sr= 16000)
        input = prepare_model_input(processed_wave, device=self.device)
        output = self.model(input)
        transcription = self.decoder(output[0].cpu())
        return transcription