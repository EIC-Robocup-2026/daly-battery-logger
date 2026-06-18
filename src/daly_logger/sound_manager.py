import io
import struct
import tempfile
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

VOICES_DIR = Path(__file__).parent.parent.parent / "voices"


def _build_wav(sample_rate: int, sample_width: int, channels: int, raw_pcm: bytes) -> bytes:
    data_size = len(raw_pcm)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_pcm)
    return buf.getvalue()


class SoundManager:
    def __init__(self, voice: str = "en_US-hfc_female-medium"):
        self._voice = voice
        self._voice_path = VOICES_DIR / f"{voice}.onnx"
        self._volume = 80
        self._player = QMediaPlayer()
        self._player.setVolume(self._volume)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.error.connect(self._on_error)
        print(f"[SOUND] Initialized with voice={voice}, volume={self._volume}")
        print(f"[SOUND] Voice path: {self._voice_path}")
        print(f"[SOUND] Voice exists: {self._voice_path.exists()}")

    def set_volume(self, volume: int):
        self._volume = max(0, min(100, volume))
        self._player.setVolume(self._volume)
        print(f"[SOUND] Volume set to {self._volume}")

    def _on_status(self, status):
        print(f"[SOUND] Media status: {status}")

    def _on_error(self, error, error_string=""):
        print(f"[SOUND] Error: {error} - {error_string}")

    def speak(self, text: str):
        if not text:
            print("[SOUND] Empty text, skipping")
            return
        print(f"[SOUND] Speaking: {text}")
        try:
            from piper import PiperVoice
            print(f"[SOUND] Piper imported OK")
            print(f"[SOUND] Loading voice from: {self._voice_path}")
            voice = PiperVoice.load(str(self._voice_path))
            print(f"[SOUND] Voice loaded OK")

            chunks = list(voice.synthesize(text))
            print(f"[SOUND] Got {len(chunks)} audio chunks")

            if not chunks:
                print("[SOUND] No audio chunks produced, skipping")
                return

            sample_rate = chunks[0].sample_rate
            sample_width = chunks[0].sample_width
            channels = chunks[0].sample_channels
            print(f"[SOUND] Audio: {sample_rate}Hz, {sample_width*8}bit, {channels}ch")

            raw_pcm = b"".join(c.audio_int16_bytes for c in chunks)
            print(f"[SOUND] Raw PCM: {len(raw_pcm)} bytes")

            wav_data = _build_wav(sample_rate, sample_width, channels, raw_pcm)
            print(f"[SOUND] WAV file: {len(wav_data)} bytes")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                tmp_path = f.name
            print(f"[SOUND] Saved to {tmp_path}")

            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(tmp_path)))
            self._player.play()
            print(f"[SOUND] Play started")
        except ImportError as e:
            print(f"[SOUND] Import error: {e}")
        except Exception as e:
            print(f"[SOUND] Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
