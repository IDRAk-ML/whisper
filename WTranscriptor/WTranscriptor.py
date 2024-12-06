import torch
import numpy as np
import soundfile as sf
from WTranscriptor.WASR import ASR
import enums
import sounddevice as sd
import math
import time
from VAD_Interface import VAD_Interface
import enums

class WTranscriptor(object):
    """
    Can take streaming raw audio blocks and convert into transcript

    This module can take the data from the sounddevice rawstreams as input. 
    It accumulates the raw audio blocks and converts them into transcripts.
    The accumulation is done based on the following conditions:
    
    1. The module keeps accumulating data, using the push function, till it reaches a duration threshold (e.g 3 secs)
    2. If a pause is detected (silence for a few seconds) the push function returns true. 
    3. If data is accumulated to more than maximum allowable duration, the push function returns true
    4. The user is expected to store the transcript at each true return of push and call the refresh function to free up memory.  
    
    
    Silence Detection/Pause Detection:
    ;For Pause Detection we are gonna use silero_vad (https://pytorch.org/hub/snakers4_silero-vad_vad/)
    ;vad_threshold is 0.6 which is obtain using different trail and erros i.e experiments
    ;After 3 seconds we are checking for speech. And Comparing the last speech end with current sample stamps
    ;If there is gap of 3 seconds we call it 3 sec pause
    
    """
    def __init__(self, config=dict()) -> None:
        """
        The constructor requires a config dictionary to set the configs. 

        All configs have a default value in case the config value is not found in the dictionary.
        A sample config is part of this module
        """
        
        self.config = config
        print(self.config)
        self.asr = ASR(config) 
        self.vad = VAD_Interface(config=config)
        self.max_allowable_duration = config.get("maximum_allowable_duration", 10) # max duration of input audio to transcribe in seconds
        self.default_allowable_duration = self.max_allowable_duration
        self.samplerate = config.get("samplerate", 16000.0)
        self.data_array = np.array([])
        self.cuda_device = config.get("cuda_device", "cpu")
        self.duration_threshold = config.get("duration_threshold", 0.5)  # after this many seconds, pass the data through the model
        self.duration_threshold_delta = config.get("duration_threshold_delta", 10) # increase in duration thresold, for next iteration. 
        
        if not "enum" in config:
            config["enum"] = dict()
        
        self.status = False
        self.transcript = None
        self.amplitude = np.iinfo(np.int16).max
        self.last_execution = time.time()
        
        # self.warmup()

    def push(self, raw_audio_block, pause_type=1, is_greedy=False, verbose=False,last_block=False,max_duration=0.0):
        """
        The main function of the Transcriptor module. For an example usage, see the code in __main__ in Transcriptor.py

        Args:
            raw_audio_block: Output from sounddevice raw stream of arbitrary blocksizey.
            
        """
        check_for_pause = True
        if max_duration != 0.0:
            self.max_allowable_duration = max_duration
            # check_for_pause = False
            # print(f"listening for {self.max_allowable_duration} hard coded, no pause will work")
        
        
        if check_for_pause:
            self.max_allowable_duration = self.default_allowable_duration
            # print(f"listening for {self.max_allowable_duration}s or pause")
        current_time = time.time()
        # print('I am inside push')
        gen_transcript = False
        #obtaining audio from bytes
        tmp_np = self.byte2np(raw_audio_block)
        pause_status = enums.NORMAL_STATUS # -1000
        self.data_array = np.hstack((self.data_array,tmp_np))
        duration  = len(self.data_array) / self.samplerate #duration 16000/16000=1s\
        speech_dict=None
        no_response_flag = False
        
        if duration >= self.duration_threshold: #if duration is larger than 3s
            # print(duration,self.duration_threshold)
            data = self.data_array
            #passing data from VAD Model
            # print(len(data) % 16000)
            if check_for_pause:
                if current_time - self.last_execution >= 0.4:
                    pause_status = self.vad.pause_status(data=self.data_array)
                    self.last_execution = current_time
                    
                if pause_status == enums.PAUSE: #if speech detected
                    print('[+] Pause Detected')
                    self.status=True
                    gen_transcript = True

                elif pause_status == enums.NO_RESPONSE:
                    self.status = True
                    gen_transcript = True
                    no_response_flag = True

                    
            if duration > self.max_allowable_duration: #10 second acheived
                print("[-] Max Limit Exceed")
                self.status = True
                gen_transcript = True
        
        if gen_transcript or last_block: #if last block of file or condition of pause or max duration meet
            # print(self.data_array)
            transcript = self.asr.get_transcript(self.data_array)

            # print(transcript,len(transcript[1]))
            self.transcript = transcript
            self.status=True
        
        if no_response_flag:
            temp_transcript = ([], '')
            if len(self.transcript[1]) < 5 and ('you' in self.transcript[1].lower()): 
                self.transcript = temp_transcript
            elif len(self.transcript[1]) < 12 and ('thank you' in self.transcript[1].lower()):
                self.transcript = temp_transcript
            else:
                self.transcript = self.transcript
            
        return self.status

    def refresh(self):
        self.data_array = np.array([])
        self.transcript = None
        self.status = False
        self.duration_threshold = self.config.get("duration_threshold", 3)

    def byte2np(self, data):
        return np.frombuffer(data, dtype='int16')


# -- For testing the module independantly
if __name__ == "__main__":

    filepath  = "audios/backy.wav"
    file_object =  sf.SoundFile(filepath)
    blocksize = 16000
    dtype = 'int16'
    samplerate = file_object.samplerate
    last_block_index = int(len(file_object)/blocksize)
    # added support for config
    config=dict()
    config = {"sample_rate":16000,"duration_threshold":3,"vad_threshold":0.6,"model_path":"base.en"}
    transcriptor = WTranscriptor(config)
    transcpt=''
    raw_data = file_object.buffer_read(blocksize, dtype=dtype)
    block_index = 1
    is_last_processing_block=False
    import timeit

    start = timeit.default_timer()
    while True:
        while (not transcriptor.push(raw_data, pause_type="small",last_block=is_last_processing_block)):
            raw_data = file_object.buffer_read(blocksize, dtype=dtype)
            block_index+=1
            if last_block_index-1 < block_index:
                is_last_processing_block=True
            else:
                is_last_processing_block=False
            
        transcpt += transcriptor.transcript[1]   
        transcriptor.refresh()   
        if is_last_processing_block:
            file_object.close()
            break
    end = timeit.default_timer()
    print('Time',end-start)
    print(transcpt) 

    
    
    
    
