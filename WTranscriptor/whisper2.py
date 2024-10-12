from faster_whisper import WhisperModel
# audio_path='/home/ali/Desktop/idrak_work/transcriptor_module-transcriptor-module/WTranscriptor/audios/preamble_5sec_resample_with_pause.wav'
import timeit
import numpy as np
# from pydub import AudioSegment
# from scipy.io.wavfile import write
import warnings
import torch
import os
import zlib
warnings.filterwarnings('ignore')

'''
Faster Implementation of Whisper
'''


vad_model, utils = torch.hub.load('snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)


(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

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
    def __init__(self,model_path='',file_processing=False,word_timestamp=True):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''

        self.model_path = model_path
        print(self.model_path)
        # device='cpu'
        # compute_type="int8"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device == "cuda" , "cuda check")
        if device == 'cuda':
            print(f"[INFO] Loading {self.model_path} on Cuda")
            try:
                self.model = WhisperModel(self.model_path, device="cuda", compute_type="int8_float16")
            except ValueError:
                print("[INFO] Cuda Support Issue Moving to CPU")
                self.model = WhisperModel(self.model_path, device="cpu", compute_type="int8")#,num_worskers=5,cpu_threads=8
        else:
            print("[INFO] Loading on CPU")
            self.model = WhisperModel(self.model_path, device="cpu", compute_type="int8",
                                      cpu_threads=8,num_workers=8)#,num_worskers=5,cpu_threads=8
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
        
        segments, info = self.model.transcribe(self.audio_path, beam_size=1,
                                               best_of=1,without_timestamps=True,language='en')
        transcription = ""
        for segment in segments:
            transcription += segment.text
        return transcription,[]
    
    
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
        print('Some thing Recieve Working on it')

        t1 = timeit.default_timer()
        speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=16000,threshold=0.1)
        print(speech_timestamps)
        if speech_timestamps:
            wave = torch.from_numpy(wave)
            wave1 = collect_chunks(speech_timestamps, wave)
            wave = wave1.numpy()
        
            beam_size=None
            best_of=3
            wave = wave / np.iinfo(np.int16).max #normalize
            t1 = timeit.default_timer()
            segments, info = self.model.transcribe(wave, beam_size=5, best_of=3,without_timestamps=True,language='en')
            transcription = ""
            for segment in segments:
                transcription += segment.text
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]


