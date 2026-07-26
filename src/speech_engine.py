"""
╔══════════════════════════════════════════════════════════════════╗
║             EchoVision  –  src/speech_engine.py                  ║
║  Non-blocking text-to-speech: sentences are queued and spoken    ║
║  in a background thread so the video/detection loop never stalls.║
╚══════════════════════════════════════════════════════════════════╝
"""

import queue
import threading

from . import config
from .utils import log
import pythoncom

class SpeechEngine(threading.Thread):
    def __init__(self, enabled=True):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=8)
        self.enabled = enabled
        self._engine = None
        self._stop_flag = False

    def run(self):
        if not self.enabled:
            return

        pythoncom.CoInitialize()

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", config.TTS_RATE)
            self._engine.setProperty("volume", config.TTS_VOLUME)

            while not self._stop_flag:
                sentence = self.q.get()
                if sentence is None:
                    break

                try:
                    print("[TTS SPEAKING]:", sentence)
                    self._engine.say(sentence)
                    self._engine.runAndWait()
                except Exception as e:
                    log.warning(f"TTS error: {e}")

        finally:
            pythoncom.CoUninitialize()

    def speak(self, sentence: str):
        if not sentence or not self.enabled:
            return
        if self.q.full():
            return   # drop — don't let stale announcements pile up
        self.q.put_nowait(sentence)

    def stop(self):
        self._stop_flag = True
        self.q.put(None)
