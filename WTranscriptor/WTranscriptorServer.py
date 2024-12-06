import torch
import numpy as np
import soundfile as sf
from WTranscriptor.WASR import ASR
import enums

#------------------- For Voice Activity Detection Model Loading --------------
#
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

(get_speech_timestamps,
 _, read_audio,VADIterator,
 *_) = utils



class WTranscriptorServer(object):
    """

    Can take streaming raw audio blocks and convert into transcript. Same as Transcriptor but works for multiple clients.
    Essentially, the difference is that instead of one data_array, we have a dictionary containing as many data_array objects as the 
    number of clients. Also, self.transcript is a dictionary containing as many transcripts as the number of clients.
    The push function now requires a client-id as well.

    This module can take the data from the sounddevice rawstreams as input. 
    It accumulates the raw audio blocks and converts them into transcripts.
    The accumulation is done based on the following conditions:
    
    1. The module keeps accumulating data, using the push function, till it reaches a duration threshold (e.g 3 secs)
    2. After the duration threshold, a transcript is generated and duration threshold is increased by delta seconds (e.g 1 sec)
    3. If a pause is detected (silence for a few seconds) the push function returns true. There are two settings for pause duration large and small
    4. If data is accumulated to more than maximum allowable duration, the push function returns true
    5. The user is expected to store the transcript at each true return of push and call the refresh function to free up memory.  

    """
    def __init__(self, config=dict()) -> None:
        """
        The constructor requires a config dictionary to set the configs. 

        All configs have a default value in case the config value is not found in the dictionary.
        A sample config is part of this module
        """
        
        config["pause_time"] = config.get("pause_time", 2) # defining it here as default because it is used by ASR as well.
        self.config = config
        self.asr = ASR(config)
        self.vad_model = vad_model
        self.vad_threshold=config.get("vad_threshold",0.6) 
        # max duration at the start of audio after which a no response is detected, should be greater than pause time
        self.max_allowable_duration = config.get("maximum_allowable_duration", 10) # max duration of input audio to transcribe in seconds
        self.samplerate = config.get("samplerate", 16000)
        self.data_array = dict() # array to store audio blocks
        self.cuda_device = config.get("cuda_device", "cpu")
        self.global_duration_offset = dict()
        if self.cuda_device == "cpu":
            print("WARNING: Running on CPU")
        self.duration_threshold = dict()   # wait till this value then pass the data through the model, dictionary keys correspond to clients
        self.duration_threshold_delta = config.get("duration_threshold_delta", 1) # increase in duration thresold, for next iteration.              
        self.transcript = dict()
        self.amplitude = np.iinfo(np.int16).max


    def push(self, raw_audio_block, client_id, emissions_only=False, last_block=False,
            is_greedy=False, verbose=False):
        """
        The main function of the Transcriptor module. For an example usage, see the code in __main__ in Transcriptor.py

        Args:
            raw_audio_block: (bytes) Output from sounddevice raw stream of arbitrary blocksize
            client_id: (str) the id of the client from whom the data is coming
            puase_type (str) : if set to small, pause will return true when a small pause is detected otherwise it will return true for large pause only.
        """
        
        # check if the client with this ID already exists or do we need to make a new array for it
        self.add_if_required(client_id)
        status = False
        gen_transcript = False
        tmp_np = self.byte2np(raw_audio_block)
        self.data_array[client_id] = np.hstack((self.data_array[client_id],tmp_np))
        duration  = len(self.data_array[client_id]) / self.samplerate 
        speech_dict=None
        if duration > self.duration_threshold[client_id]:
            data = self.data_array[client_id]

            if len(data) % 16000 == 0: #running only when a sec ticks
                speech_dict = get_speech_timestamps(self.data_array[client_id], vad_model, sampling_rate=int(self.samplerate),threshold=self.vad_threshold)

            self.duration_threshold[client_id] = self.duration_threshold[client_id] + self.duration_threshold_delta
            if speech_dict: #if speech detected
                max_end = max(speech_dict, key=lambda x:x['end'])  #checking the end of speech
                #if current data frame and last speech index gap is larger than 48000 i.e. 3 sec
                if ((len(data)-max_end['end'])/self.samplerate) >= self.duration_threshold[client_id]: #small pause detected;
                    
                    print('[+] Pause Detected')
                    status=True
                    gen_transcript = True
                
            if duration > self.max_allowable_duration: #10 second acheived
                print("[-] Max Limit Exceed")
                status = True
                gen_transcript = True

            if gen_transcript or last_block: #if last block of file or condition of pause or max duration meet
                transcript = self.asr.get_transcript(self.data_array[client_id])
                self.transcript[client_id] = transcript
                status = True
            
        return status


    def refresh(self, client_id, reset_global_offset=True):
        self.data_array[client_id] = np.array([])
        self.transcript[client_id] = None
        self.duration_threshold[client_id] = self._get_initial_duration_threshold()
        if reset_global_offset:
            self.global_duration_offset[client_id] = 0
        

    def hard_refresh(self):
        self.data_array = dict()
        self.transcript = dict()
        self.duration_threshold = dict()
        self.conversation_flags = dict()
        self.global_duration_offset = dict()
        
    def add_if_required(self, client_id):
        """
        Makes an addition to the self.data_array dict and self.transcript dict if the client_id is being used for the first time; otherwise pass
        """
        if client_id not in self.data_array.keys():
            self.refresh(client_id)
    

    def _get_initial_duration_threshold(self):
        return 3
    
    
    def byte2np(self, data):
        return np.frombuffer(data, dtype='int16')

    
    def warmup(self):
        self.hard_refresh()

    


