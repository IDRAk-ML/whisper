# Configuration for ASR
import os

model_path = os.getenv('MODEL_NAME','whisper-medium.en')

config = {
    "sample_rate": 16000,
    "duration_threshold": 3,
    "vad_threshold": 0.3,
    "model_path": model_path,
    'mac_device': True,
    'model_name': 'whisper',
    'enable_vad': True,
    'vad_thresold': 0.3,
    'type': '',
}
def str_to_bool(value):
    true_values = {'true', '1', 'yes', 'on', 'y', 't'}
    false_values = {'false', '0', 'no', 'off', 'n', 'f'}

    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError("Input must be a string or boolean.")

    val = value.strip().lower()
    if val in true_values:
        return True
    elif val in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean.")


if config['type'] == 'faster_whisper':
    BASE_PATH = ''
else:
    BASE_PATH = 'openai/'
config['model_path'] = BASE_PATH + config['model_path']


HELPING_ASR_FLAG = True
HELPING_ASR_MODEL = {'model_list':['whisper_at','sensevoice'],'selected_model':0}
SMART_AM_CHECK = os.getenv('SMART_AM_CHECK',"ON")
SMART_AM_CHECK = str_to_bool(SMART_AM_CHECK)


WHISPER_AT_SERVER_URL = 'localhost' # if not docker set tup localhost

WHISPER_AT_ADDRESS = os.getenv('WHISPER_AT_ADDRESS','localhost:9007')
AMD_SERVER_ADDRESS = os.getenv('AMD_SERVER_ADDRESS','localhost:8034')
ENV_DOCKER = True
AMD_DOCKER_NETWORK = AMD_SERVER_ADDRESS
WHISPER_AT_DOCKER_NETWORK = WHISPER_AT_ADDRESS