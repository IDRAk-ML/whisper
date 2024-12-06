#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from WTranscriptor.w2as import WhisperTranscriptorAPI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=0.5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=10,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
    
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = WhisperTranscriptorAPI(model_path=model)

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    phrase_start_time = None
    phrase_duration = 0.0
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not phrase_start_time:
                phrase_start_time = now
            
            if not data_queue.empty():
                phrase_complete = False

                elapsed_time = (now - phrase_start_time).total_seconds()
                phrase_duration += elapsed_time
                if phrase_duration >= 10.0:  # 10 seconds of continuous speech
                    phrase_complete = True
                    print('10 seconds of continuous speech')
                    phrase_duration = 0.0  # reset the counter
                
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                    phrase_duration = 0.0  # reset the counter
                    print('Pause Detected')
                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = audio_model.generate_transcript(audio_path=temp_file)
                text = result[0].strip()

                if phrase_complete:
                    transcription.append(text)
                    phrase_start_time = None  # start counting a new phrase

                for line in transcription:
                    print(line)
                sleep(0.1)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
