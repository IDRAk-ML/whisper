# Configuration for ASR
config = {
    "sample_rate": 16000,
    "duration_threshold": 3,
    "vad_threshold": 0.3,
    "model_path": "whisper-medium.en",
    'mac_device': True,
    'model_name': 'whisper',
    'enable_vad': True,
    'vad_thresold': 0.3,
    'type': '',
}

if config['type'] == 'faster_whisper':
    BASE_PATH = ''
else:
    BASE_PATH = 'openai/'
config['model_path'] = BASE_PATH + config['model_path']


HELPING_ASR_FLAG = True
HELPING_ASR_MODEL = {'model_list':['whisper_at','sensevoice'],'selected_model':0}
SMART_AM_CHECK = False

WHISPER_AT_SERVER_URL = 'localhost'