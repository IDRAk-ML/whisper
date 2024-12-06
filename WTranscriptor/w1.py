#ali's whisper
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torchaudio
import os
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
import warnings
import timeit
import whisper
from stable_whisper import modify_model
# import whisper_timestamped as whisper
warnings.filterwarnings('ignore')


"""
**WhisperTranscriptorAPI Defination**
"""

class WhisperTranscriptorAPI:
    '''
    This Module is based on OpenAI Whisper for Audio Transcription.
    We need WhisperProcessor and WhisperConditionalGeneration for 
    CTC task i.e. ASR. 
    example:
          whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-tiny.en')
          
    '''
    #----------------------- constructor -------------------------------------
    #
    def __init__(self,model_path='openai/whisper-tiny.en',file_processing=False,word_timestamp=True):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''

        self.model_path = 'openai/whisper-tiny.en'
        self.processor = WhisperProcessor.from_pretrained(self.model_path) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        if file_processing:
            self.model = whisper.load_model('base.en').to(self.device)
            modify_model(self.model)
            # print(self.model.device)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            # print(self.model.device)
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
        self.OUTPUT_DIR= "audios"


    #--------------------------- convert mp3 to wav ---------------------------
    #

    def convert_to_wav(self,filepath=''):
        '''
        If file is mp3 first convert it to wav.
        
        1) Read audio segment from Path
        2) Sample the audio to 16kHZ
        3) Save file in temp
        args:
            filepath(str): path of audio file example audio.mp3
        returns:
            path of temporary generated wav file
        '''

        path =filepath
        filename=os.path.basename(filepath) #Getting filename from path
        save_path = f"{self.OUTPUT_DIR}"
        if not os.path.exists(save_path): #If path not exist create a directory
            os.makedirs(save_path, exist_ok=True)
        if os.path.exists(save_path):
            try:
                sound = AudioSegment.from_mp3(path) #Read a sound file
                # print('Working Here')
                sound = sound.set_frame_rate(16000) #resample to 16000
                sound.export(f"{save_path}/{filename[:-4]}.wav", format="wav") #save to *.wav
                return f"{save_path}/{filename[:-4]}.wav"
            except Exception as e:
                print(path, e)


    #------------------------- genrate transcript -----------------------------
    #
    def generate_transcript(self, audio_path=''):
        
        '''
        Generate the transcript from audio file using Whisper
        1) This function will read the .wav audio using torchaudio
        2) Process the audio using WhisperProcessor
        3) Generate the predicted token ids using Whisper Model
        4) Decode the output using Language model
        args: 
            audio_path (str) : path of wav file
        returns:
            transcription(str): generated script
        '''
        self.audio_path = audio_path
        exten = os.path.splitext(self.audio_path)[1]
        if exten == '.mp3':
            self.audio_path = self.convert_to_wav(self.audio_path)
        print(self.audio_path)
        wave,fs=torchaudio.load(self.audio_path)
        #tensor to numpy
        wave=wave.numpy()
        if len(wave.shape)>=2:
            wave=wave[0,:]
        inputs = self.processor(wave, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        # print(input_features)
        with torch.inference_mode():
            generated_ids = self.model.generate(inputs=input_features)
        #decode the transcript using language model from processor 
        print(help(self.processor.batch_decode))
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription,generated_ids

    #----------------------- For Audio More than 30 s -------------------------
    #
    def generate_on_longer_file(self,audio_path=''):
        '''
        Whisper only work on audio chunk of 30s. If Audio is longer than 30s it 
        turncate it to 30 s. In other case if audio is lesser than 30s it pad
        uo to 30s then generate trabscript. Here we are using transcribe function

        args:
            audio_path (str): path to Audio file
        returns:
            transcript (str): generate transcript over audio file
        '''
        self.audio_path = audio_path
        exten = os.path.splitext(self.audio_path)[1]
        if exten == '.mp3':
            self.audio_path = self.convert_to_wav(self.audio_path)
        result = self.model.transcribe(self.audio_path,language='en')['text']
        return result
    
    #------------------------ For Generating the Timestamps -------------------
    #
    def generate_timestamps(self,audio_path=''):
        '''
        Whisper model can return the sentene base timestamps rather than word 
        level timestamps as Wave2Vec models returns based on phonames. There
        are two solutions for word level timestamps 
        1) whisperX which aligns the output of Whisper with Wave2Vec 
        2) Attention based word to timestamp mapping. whisper_timestamp



        args:
            audio_path (str): path to Audio file
        returns:
            transcript (str): generate sentence based timestamps for audio
        '''
        self.audio_path = audio_path
        exten = os.path.splitext(self.audio_path)[1]
        if exten == '.mp3':
            self.audio_path = self.convert_to_wav(self.audio_path)
        result = self.model.transcribe(self.audio_path,language='en')
        return result
    #---------------------------- save audio-----------------------------------
    #
    def save_audio(self,wave):
        
        '''
        Save the audio file for testing
        '''
        samplerate = 16000
        write("exame1.wav", samplerate, wave.astype(np.int16))
    #-------------------- generate transcript from nmpy array ----------------
    #
    
    def generate_transcript_numpy(self, wave):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''
        wave = wave / np.iinfo(np.int16).max #normalize
        t1 = timeit.default_timer()
        inputs = self.processor(wave, return_tensors="pt",sampling_rate=16000) #tokenize
        input_features = inputs.input_features.to(self.device)
        # print(input_features.to(self.device))
        with torch.inference_mode():
            print(help(self.model.generate))
            generated_ids = self.model.generate(inputs=input_features)
        #decode the transcript using language model from processor 
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        t2 = timeit.default_timer()
        print('Time taking for response',t2-t1)
        print('Audio Length',len(wave)/16000)
        return transcription,generated_ids
    
    def generate_features_only(self,wave):
        '''
        Generate the features IDs only. 
        No decoding is performed here
        '''
        wave = wave / np.iinfo(np.int16).max #normalize
        inputs = self.processor(wave, return_tensors="pt",sampling_rate=16000) #tokenize
        input_features = inputs.input_features.to(self.device)
        # print(input_features.to(self.device))
        with torch.inference_mode():
            generated_ids = self.model.generate(inputs=input_features)
        return generated_ids

if __name__ == "__main__":
    """Model Initialization"""

    import timeit
    # Please set file_processing to True when you have to run on longer file other wise use old apporch
    whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-base.en',file_processing=True)

    """Experiments:"""
    t1 = timeit.default_timer()
    transcript = whisper_transcriptor.generate_on_longer_file(audio_path='/home/ali/Desktop/idrak_work/transcriptor_module-transcriptor-module/WTranscriptor/audios/amy.wav')
    t2 = timeit.default_timer()
    print(transcript)
    print(t2-t1)
  
