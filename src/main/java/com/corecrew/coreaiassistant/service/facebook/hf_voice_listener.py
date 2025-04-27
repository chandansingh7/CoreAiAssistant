#!/usr/bin/env python3
import signal, queue, threading, time, os, warnings, contextlib, sys
import sounddevice as sd
import numpy as np
import webrtcvad
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, logging as hf_logging
import torch
from pyctcdecode import build_ctcdecoder

# ─── QUIET WARNINGS & LOGS ─────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_NAME    = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
# Resolve LM_PATH relative to this script’s location
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
LM_PATH       = os.path.join(SCRIPT_DIR, "4gram.bin")
SAMPLE_RATE   = 16000
CHANNELS      = 1
FRAME_DUR     = 0.030
VAD_MODE      = 3
MIN_SPEECH_MS = 400
MIN_SILENCE_MS= 600
_ms_per_frame = FRAME_DUR * 1000
MIN_SF        = int(MIN_SPEECH_MS  / _ms_per_frame)
MIN_SL        = int(MIN_SILENCE_MS / _ms_per_frame)

# ─── PICK MIC (silent) ──────────────────────────────────────────────────────────
devices    = sd.query_devices()
input_idxs = [i for i,d in enumerate(devices) if d["max_input_channels"] > 0]
if not input_idxs:
    raise RuntimeError("No input device found!")
dev_idx    = input_idxs[0]

# ─── CHECK LM EXISTS ────────────────────────────────────────────────────────────
if not os.path.isfile(LM_PATH):
    print(f"❌ Language model not found at {LM_PATH}", file=sys.stderr)
    sys.exit(1)

# ─── LOAD & WARM UP (fully silent) ──────────────────────────────────────────────
with open(os.devnull, "w") as devnull, \
        contextlib.redirect_stdout(devnull), \
        contextlib.redirect_stderr(devnull):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to("cpu")
    vocab     = list(processor.tokenizer.get_vocab().keys())
    decoder   = build_ctcdecoder(vocab, LM_PATH)
    # warm up
    vals = processor(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_values
    with torch.no_grad():
        model(vals)

# ─── VAD & QUEUE ────────────────────────────────────────────────────────────────
vad       = webrtcvad.Vad(VAD_MODE)
audio_q   = queue.Queue()
terminate = False

def on_sig(sig, frame):
    global terminate
    terminate = True
signal.signal(signal.SIGINT,  on_sig)
signal.signal(signal.SIGTERM, on_sig)

def audio_cb(indata, frames, time_info, status):
    audio_q.put(indata[:,0].tobytes())

def recognize_loop():
    buf = []; sf = sl = 0; listening = False; last_txt = ""
    while not terminate:
        try:
            pcm = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        is_sp = vad.is_speech(pcm, SAMPLE_RATE)
        if not listening:
            if is_sp:
                listening, buf, sf, sl = True, [pcm], 1, 0
        else:
            buf.append(pcm)
            sf, sl = (sf+1, 0) if is_sp else (sf, sl+1)
            if sf >= MIN_SF and sl >= MIN_SL:
                raw = b"".join(buf)
                buf, listening, sf, sl = [], False, 0, 0
                audio = (np.frombuffer(raw, dtype=np.int16)
                         .astype(np.float32)/32768.0)
                # CTC + LM decode (beam_width=100)
                iv = processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt"
                ).input_values
                with torch.no_grad():
                    logits = model(iv).logits[0].cpu().numpy()
                txt = decoder.decode(logits, beam_width=100).strip()
                if len(txt) >= 3 and txt != last_txt:
                    print(txt)
                    last_txt = txt

threading.Thread(target=recognize_loop, daemon=True).start()

# ─── START STREAM ───────────────────────────────────────────────────────────────
with sd.InputStream(
        device     = dev_idx,
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = "int16",
        blocksize  = int(FRAME_DUR * SAMPLE_RATE),
        latency    = "low",
        callback   = audio_cb,
):
    while not terminate:
        time.sleep(0.1)
