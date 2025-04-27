#!/usr/bin/env python3
import os
import json
import queue
import threading
import sounddevice as sd
from vosk import SetLogLevel, Model, KaldiRecognizer
import zipfile
import urllib.request
import signal
import time

# ─── SILENCE VOSK / KALDI LOGS ───────────────────────────────────────────────
SetLogLevel(-1)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS    = 1
BLOCK_SEC   = 0.2  # 200 ms blocks

# ─── PATHS & MODEL DOWNLOAD ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "vosk-model-en-us-0.22")
MODEL_ZIP  = MODEL_DIR + ".zip"
MODEL_URL  = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"

if not os.path.isdir(MODEL_DIR):
    print("⏳ Downloading Vosk model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP)
    print("⏳ Unpacking model…")
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(SCRIPT_DIR)
    os.remove(MODEL_ZIP)
    print(f"✔️ Model ready at {MODEL_DIR}")

# ─── LOAD VOSK MODEL ───────────────────────────────────────────────────────
vosk_model = Model(MODEL_DIR)
rec        = KaldiRecognizer(vosk_model, SAMPLE_RATE)
rec.SetMaxAlternatives(0)
rec.SetWords(True)

# ─── SELECT FIRST MIC DEVICE ────────────────────────────────────────────────
devices    = sd.query_devices()
input_idxs = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
if not input_idxs:
    raise RuntimeError("No input device found!")
sd.default.device = (input_idxs[0], None)

# ─── GRACEFUL TERMINATION SETUP ─────────────────────────────────────────────
terminate = False
def term_handler(signum, frame):
    global terminate
    terminate = True

signal.signal(signal.SIGINT,  term_handler)
signal.signal(signal.SIGTERM, term_handler)

# ─── AUDIO QUEUE & CALLBACK ─────────────────────────────────────────────────
audio_q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Stream status:", status)
    # indata is int16 numpy array, single channel
    pcm = indata[:,0].tobytes()
    audio_q.put(pcm)

# ─── RECOGNITION WORKER ─────────────────────────────────────────────────────
def recognition_worker():
    while True:
        pcm = audio_q.get()
        if rec.AcceptWaveform(pcm):
            raw = rec.Result()
            j   = json.loads(raw)
            # j["result"] is list of {word, start, end}
            words = [w["word"] for w in j.get("result", [])]
            text  = " ".join(words).strip()
            if text:
                print("🗣️", text)

# start recognition thread
threading.Thread(target=recognition_worker, daemon=True).start()

# ─── START STREAM & WAIT UNTIL TERMINATION ─────────────────────────────────
print("✅ Listening—speak when you're ready")
with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * BLOCK_SEC),
        dtype="int16",
        channels=1,
        callback=audio_callback,
        latency="low"
):
    while not terminate:
        time.sleep(0.1)

print("👋 Exiting, microphone closed.")
