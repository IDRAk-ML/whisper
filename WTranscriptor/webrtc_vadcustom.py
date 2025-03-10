import wave
import webrtcvad
from pydub import AudioSegment

class WebRTCVADSpeechDetector:
    def __init__(self, aggressiveness=3, frame_duration_ms=20):
        """
        Initialize the VAD speech detector.
        
        :param aggressiveness: VAD aggressiveness level (0-3)
        :param frame_duration_ms: Frame duration in ms (10, 20, or 30)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms

    def convert_to_16bit_pcm(self, file_path):
        """Convert audio to 16-bit PCM if necessary."""
        audio = AudioSegment.from_wav(file_path)
        if audio.sample_width != 2:  # Convert if not 16-bit PCM
            audio = audio.set_sample_width(2)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.channels != 1:
            audio = audio.set_channels(1)
        # output_path = file_path.replace(".wav", "_16bit.wav")
        audio.export(file_path, format="wav")
        # return output_path
        return file_path

    def read_audio(self, file_path):
        """Read a WAV audio file and return raw audio bytes and sample rate."""
        file_path = self.convert_to_16bit_pcm(file_path)  # Ensure correct format
        with wave.open(file_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            audio_bytes = wav_file.readframes(wav_file.getnframes())
        return audio_bytes, sample_rate
    
    def process_audio(self, audio_bytes, sample_rate):
        """Process audio to detect speech segments."""
        frame_bytes = int(sample_rate * (self.frame_duration_ms / 1000.0)) * 2  # 2 bytes per sample (16-bit)
        speech_flags = []
        
        for offset in range(0, len(audio_bytes), frame_bytes):
            frame = audio_bytes[offset: offset + frame_bytes]
            if len(frame) < frame_bytes:
                break  # Drop the last partial frame
            is_speech = self.vad.is_speech(frame, sample_rate)
            speech_flags.append(is_speech)
        
        return self._extract_segments(speech_flags)
    
    def _extract_segments(self, speech_flags):
        """Post-process detected speech frames into speech segments with timestamps."""
        segments = []
        frame_sec = self.frame_duration_ms / 1000.0
        current_seg = None
        
        for i, has_speech in enumerate(speech_flags):
            t_start = i * frame_sec
            t_end = t_start + frame_sec
            
            if has_speech:
                if current_seg is None:
                    current_seg = [t_start, t_end]
                else:
                    current_seg[1] = t_end
            else:
                if current_seg:
                    segments.append(tuple(current_seg))
                    current_seg = None
        
        if current_seg:
            segments.append(tuple(current_seg))
        
        return segments
    
    def detect_speech(self, file_path):
        """Detect speech segments from a WAV file and return timestamps."""
        audio_bytes, sample_rate = self.read_audio(file_path)
        return self.process_audio(audio_bytes, sample_rate)

# Example usage:
if __name__ == "__main__":
    detector = WebRTCVADSpeechDetector(aggressiveness=3, frame_duration_ms=20)
    speech_segments = detector.detect_speech("/Users/ali/Desktop/ali_works/whisper/WTranscriptor/audios/amy.wav")
    
    for start, end in speech_segments:
        print(f"Speech from {start:.2f}s to {end:.2f}s")
