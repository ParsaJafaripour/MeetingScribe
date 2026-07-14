#!/usr/bin/env python3
"""
MeetingScribe v2 — local meeting transcription with speaker labels, for macOS.

Everything runs on your own machine. The only network use is a one-time
download of open-source model weights (Whisper from Hugging Face's CDN,
speaker models from GitHub). No accounts, no API keys, no audio ever leaves
this Mac.

Launch with:  ./venv/bin/python app.py   (or double-click "Start MeetingScribe.command")
"""

import os
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file

# ---------------------------------------------------------------- config ----

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
OUTPUT_ROOT = Path.home() / "Documents" / "MeetingScribe"
PREFERRED_PORT = 8756

MODELS = {
    "base":           "base · 74 MB · fastest, rough drafts",
    "small":          "small · 244 MB · balanced (recommended)",
    "medium":         "medium · 1.5 GB · high accuracy",
    "large-v3-turbo": "large-v3-turbo · 1.6 GB · best accuracy",
}
DEFAULT_MODEL = "small"

# Speaker models: pyannote segmentation-3.0 (MIT) + NeMo TitaNet-small embeddings,
# both converted to ONNX and hosted on the sherpa-onnx GitHub releases.
SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/nemo_en_titanet_small.onnx")  # (typo is in the real URL)
SEG_PATH = MODELS_DIR / "pyannote-segmentation-3-0.onnx"
EMB_PATH = MODELS_DIR / "titanet-small-embedding.onnx"

N_THREADS = max(1, min(4, os.cpu_count() or 2))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8 GB uploads

jobs = {}
jobs_lock = threading.Lock()
work_lock = threading.Lock()      # one heavy job at a time
_whisper_cache = {}
_diarizer_cache = {}

LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "fa": "Persian", "tr": "Turkish",
    "pl": "Polish", "sv": "Swedish", "uk": "Ukrainian", "he": "Hebrew", "ur": "Urdu",
}


# ---------------------------------------------------------------- helpers ---

def lang_name(code):
    if not code:
        return "unknown"
    return LANG_NAMES.get(code, code.upper())


def fmt_clock(seconds, force_hours=False):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h or force_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_stem(name):
    stem = Path(name).stem
    stem = re.sub(r"[^\w\- ]+", "", stem).strip() or "recording"
    return stem[:60]


def get_whisper(name):
    if name not in _whisper_cache:
        from faster_whisper import WhisperModel
        _whisper_cache[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _whisper_cache[name]


def download_file(url, dest, job, label):
    """Download url -> dest with progress reported into the job's tail."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MeetingScribe/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                job["tail"] = f"{label}: {done / 1048576:.0f} / {total / 1048576:.0f} MB"
                job["progress"] = done / total
            else:
                job["tail"] = f"{label}: {done / 1048576:.0f} MB"
    os.replace(tmp, dest)


def ensure_speaker_models(job):
    """Fetch speaker models once; afterwards fully offline."""
    if SEG_PATH.exists() and EMB_PATH.exists():
        return
    job["phase"] = "downloading-speaker-models"
    job["progress"] = 0.0
    if not SEG_PATH.exists():
        tarball = MODELS_DIR / "segmentation.tar.bz2"
        download_file(SEG_URL, tarball, job, "Speaker segmentation model")
        with tarfile.open(tarball, "r:bz2") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("model.onnx"))
            member.name = SEG_PATH.name
            tf.extract(member, MODELS_DIR)
        tarball.unlink(missing_ok=True)
    if not EMB_PATH.exists():
        download_file(EMB_URL, EMB_PATH, job, "Speaker voice model")


def get_diarizer(num_speakers):
    key = int(num_speakers)
    if key not in _diarizer_cache:
        import sherpa_onnx
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SEG_PATH)),
                num_threads=N_THREADS,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMB_PATH), num_threads=N_THREADS),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=key, threshold=0.5),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        _diarizer_cache[key] = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _diarizer_cache[key]


def assign_speakers(segments, diar):
    """Give each Whisper segment the diarization speaker with the most overlap,
    then relabel speakers 1..N in order of first appearance."""
    for seg in segments:
        best, best_ov = None, 0.0
        mid = (seg["start"] + seg["end"]) / 2
        nearest, nearest_d = None, float("inf")
        for d in diar:
            ov = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if ov > best_ov:
                best_ov, best = ov, d["speaker"]
            dist = abs((d["start"] + d["end"]) / 2 - mid)
            if dist < nearest_d:
                nearest_d, nearest = dist, d["speaker"]
        seg["speaker"] = best if best is not None else nearest

    prev = None
    for seg in segments:
        if seg["speaker"] is None:
            seg["speaker"] = prev
        prev = seg["speaker"]
    nxt = None
    for seg in reversed(segments):
        if seg["speaker"] is None:
            seg["speaker"] = nxt if nxt is not None else 0
        nxt = seg["speaker"]

    order, mapping = [], {}
    for seg in segments:
        if seg["speaker"] not in mapping:
            mapping[seg["speaker"]] = len(order) + 1
            order.append(seg["speaker"])
        seg["speaker"] = mapping[seg["speaker"]]
    return len(order)


def build_turns(segments):
    """Merge consecutive same-speaker segments into speaker turns."""
    turns = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg["end"]
        else:
            turns.append({"speaker": seg["speaker"], "start": seg["start"],
                          "end": seg["end"], "text": text})
    return turns


def build_paragraphs(segments, gap=2.0, max_len=700):
    paras, cur, prev_end = [], "", None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        new_para = False
        if prev_end is not None and seg["start"] - prev_end >= gap:
            new_para = True
        if len(cur) > max_len and cur.rstrip().endswith((".", "!", "?")):
            new_para = True
        if new_para and cur:
            paras.append(cur.strip())
            cur = ""
        cur = (cur + " " + text).strip()
        prev_end = seg["end"]
    if cur:
        paras.append(cur.strip())
    return paras


def write_outputs(job):
    stem = safe_stem(job["filename"])
    stamp = datetime.now().strftime("%Y-%m-%d %H.%M")
    outdir = OUTPUT_ROOT / f"{stem} — {stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    segments = job["segments"]
    duration = fmt_clock(job["duration"], force_hours=True)
    lang_line = lang_name(job.get("language"))
    if job.get("language_probability"):
        lang_line += f" ({job['language_probability']:.0%} confidence)"
    transcribed_at = datetime.now().strftime("%A, %B %-d, %Y at %-I:%M %p")

    labeled = bool(job.get("turns"))
    meta = (f"File: {job['filename']} · Duration: {duration} · Language: {lang_line} · "
            f"Model: Whisper {job['model']} (ran locally)")
    if labeled:
        meta += f" · Speakers detected: {job['num_speakers']}"

    if labeled:
        turns = job["turns"]
        body_lines = [f"**Speaker {t['speaker']}** — {fmt_clock(t['start'])}\n{t['text']}"
                      for t in turns]
        plain = [f"Speaker {t['speaker']}: {t['text']}" for t in turns]
        md = (f"# Meeting transcript — {job['filename']}\n\n{meta}\n"
              f"Transcribed: {transcribed_at}\n\n---\n\n## Transcript\n\n"
              + "\n\n".join(body_lines)
              + "\n\n---\n\n## Plain text\n\n" + "\n\n".join(plain) + "\n")
        txt = "\n\n".join(plain) + "\n"
        srt_iter = ((f"Speaker {s['speaker']}: {s['text'].strip()}", s)
                    for s in segments if s["text"].strip())
    else:
        ts_lines = [f"[{fmt_clock(s['start'])}] {s['text'].strip()}"
                    for s in segments if s["text"].strip()]
        paragraphs = build_paragraphs(segments)
        md = (f"# Meeting transcript — {job['filename']}\n\n{meta}\n"
              f"Transcribed: {transcribed_at}\n\n---\n\n## Timestamped transcript\n\n"
              + "\n\n".join(ts_lines)
              + "\n\n---\n\n## Plain text\n\n" + "\n\n".join(paragraphs) + "\n")
        txt = "\n\n".join(paragraphs) + "\n"
        srt_iter = ((s["text"].strip(), s) for s in segments if s["text"].strip())

    srt_parts = []
    for i, (text, s) in enumerate(srt_iter, start=1):
        srt_parts.append(f"{i}\n{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}\n{text}\n")
    srt = "\n".join(srt_parts)

    files = {}
    for ext, content in (("md", md), ("txt", txt), ("srt", srt)):
        p = outdir / f"transcript.{ext}"
        p.write_text(content, encoding="utf-8")
        files[ext] = str(p)

    job["files"] = files
    job["outdir"] = str(outdir)
    job["markdown"] = md


def run_job(job_id):
    job = jobs[job_id]
    try:
        with work_lock:
            # 1. Whisper model (may download once)
            job["phase"] = "loading-model"
            model = get_whisper(job["model"])

            # 2. Decode the recording once; reuse for diarization + transcription
            job["phase"] = "decoding"
            from faster_whisper import decode_audio
            samples = decode_audio(job["audio_path"], sampling_rate=16000)
            job["duration"] = len(samples) / 16000.0

            # 3. Speaker diarization (optional, with graceful fallback)
            diar = None
            if job["diarize"]:
                try:
                    ensure_speaker_models(job)
                    job["phase"] = "diarizing"
                    job["progress"] = 0.0
                    diarizer = get_diarizer(job["num_speakers_req"])

                    def on_progress(done, total):
                        job["progress"] = done / max(1, total)
                        return 0

                    result = diarizer.process(samples, callback=on_progress)
                    diar = [{"start": s.start, "end": s.end, "speaker": s.speaker}
                            for s in result.sort_by_start_time()]
                    if not diar:
                        raise RuntimeError("no speech regions found")
                except Exception as e:
                    diar = None
                    job["warning"] = f"Speaker labeling failed ({e}); transcript saved without labels."

            # 4. Transcription
            job["phase"] = "transcribing"
            job["progress"] = 0.0
            segments_iter, info = model.transcribe(
                samples, beam_size=5, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            job["language"] = info.language
            job["language_probability"] = info.language_probability
            for seg in segments_iter:
                job["segments"].append({"start": seg.start, "end": seg.end, "text": seg.text})
                if job["duration"]:
                    job["progress"] = min(0.999, seg.end / job["duration"])
                job["tail"] = seg.text.strip()

            # 5. Merge + write files
            if diar and job["segments"]:
                job["num_speakers"] = assign_speakers(job["segments"], diar)
                job["turns"] = build_turns(job["segments"])

        write_outputs(job)
        job["progress"] = 1.0
        job["phase"] = "done"
    except Exception as e:
        job["phase"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(job["audio_path"])
        except OSError:
            pass


# ----------------------------------------------------------------- routes ---

@app.get("/")
def index():
    return PAGE


@app.post("/transcribe")
def transcribe():
    f = request.files.get("audio")
    if f is None or not f.filename:
        return jsonify({"error": "No file received."}), 400
    model = request.form.get("model", DEFAULT_MODEL)
    if model not in MODELS:
        model = DEFAULT_MODEL
    diarize = request.form.get("diarize") == "1"
    try:
        num_speakers = int(request.form.get("speakers", -1))
    except ValueError:
        num_speakers = -1
    if not (2 <= num_speakers <= 12):
        num_speakers = -1  # auto

    suffix = Path(f.filename).suffix or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()

    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "phase": "queued", "progress": 0.0, "tail": "", "error": None, "warning": None,
            "filename": f.filename, "model": model, "audio_path": tmp.name,
            "diarize": diarize, "num_speakers_req": num_speakers, "num_speakers": 0,
            "segments": [], "turns": None, "duration": 0,
        }
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    out = {
        "phase": job["phase"],
        "progress": job["progress"],
        "tail": job["tail"],
        "error": job["error"],
        "warning": job["warning"],
        "processed": fmt_clock(job["progress"] * job["duration"], force_hours=True),
        "total": fmt_clock(job["duration"], force_hours=True) if job["duration"] else "–:––:––",
    }
    if job["phase"] == "done":
        out.update({
            "outdir": job["outdir"],
            "language": lang_name(job.get("language")),
            "markdown": job["markdown"],
            "segments": len(job["segments"]),
            "speakers": job["num_speakers"],
        })
    return jsonify(out)


@app.get("/download/<job_id>/<ext>")
def download(job_id, ext):
    job = jobs.get(job_id)
    if not job or job["phase"] != "done" or ext not in job["files"]:
        return jsonify({"error": "Not available."}), 404
    stem = safe_stem(job["filename"])
    return send_file(job["files"][ext], as_attachment=True,
                     download_name=f"{stem} transcript.{ext}")


@app.post("/reveal/<job_id>")
def reveal(job_id):
    job = jobs.get(job_id)
    if not job or job["phase"] != "done":
        return jsonify({"error": "Not available."}), 404
    subprocess.Popen(["open", job["outdir"]])
    return jsonify({"ok": True})


# ------------------------------------------------------------------- page ---

MODEL_OPTIONS = "\n".join(
    f'<option value="{k}"{" selected" if k == DEFAULT_MODEL else ""}>{v}</option>'
    for k, v in MODELS.items()
)
SPEAKER_OPTIONS = '<option value="-1" selected>Speakers: auto</option>' + "\n".join(
    f'<option value="{n}">Speakers: {n}</option>' for n in range(2, 9)
)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MeetingScribe</title>
<style>
  :root{
    --chassis:#191816; --chassis-2:#211f1c;
    --panel:#efece4; --panel-2:#e6e2d7; --line:#d4cfc1;
    --ink:#22201c; --muted:#7a7467;
    --amber:#b97f1f; --amber-bright:#d99a2b;
    --ok:#3d6b4f; --err:#a03d2e;
    --mono:"SF Mono", ui-monospace, Menlo, monospace;
  }
  *{box-sizing:border-box; margin:0}
  html,body{height:100%}
  body{
    background:var(--chassis);
    background-image:radial-gradient(ellipse at 50% -10%, #26241f 0%, var(--chassis) 60%);
    color:var(--ink);
    font:15px/1.55 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    display:flex; justify-content:center; padding:48px 20px 64px;
  }
  .unit{width:100%; max-width:680px}

  header{display:flex; align-items:baseline; justify-content:space-between; padding:0 6px 14px}
  .brand{color:#e8e4da; font-weight:600; letter-spacing:.16em; font-size:13px; text-transform:uppercase}
  .brand em{color:var(--amber-bright); font-style:normal}
  .local{display:flex; align-items:center; gap:7px; color:#9a948a; font-family:var(--mono); font-size:11px; letter-spacing:.08em}
  .dot{width:7px; height:7px; border-radius:50%; background:var(--amber-bright)}
  @media (prefers-reduced-motion: no-preference){
    .dot.rec{animation:blink 1.6s ease-in-out infinite}
    @keyframes blink{50%{opacity:.25}}
  }

  .panel{
    background:var(--panel); border-radius:14px; padding:26px 28px;
    box-shadow:0 1px 0 rgba(255,255,255,.06) inset, 0 18px 40px rgba(0,0,0,.45);
    border-top:1px solid #f7f5ef;
  }
  .panel + .panel{margin-top:14px}

  .drop{
    border:1.5px dashed #b9b2a2; border-radius:10px; background:var(--panel-2);
    padding:34px 20px; text-align:center; cursor:pointer;
    transition:border-color .15s, background .15s;
  }
  .drop:hover,.drop.over{border-color:var(--amber); background:#e9e4d6}
  .drop h2{font-size:16px; font-weight:600; margin-bottom:4px}
  .drop p{color:var(--muted); font-size:13px}
  .drop .file{font-family:var(--mono); font-size:13px; color:var(--ink); margin-top:2px; word-break:break-all}

  .controls{display:flex; gap:12px; align-items:center; margin-top:18px; flex-wrap:wrap}
  select{
    appearance:none; -webkit-appearance:none;
    background:var(--panel-2) url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6"><path d="M1 1l4 4 4-4" stroke="%237a7467" stroke-width="1.5" fill="none"/></svg>') no-repeat right 12px center;
    border:1px solid var(--line); border-radius:8px; padding:10px 32px 10px 12px;
    font:13px var(--mono); color:var(--ink); cursor:pointer;
  }
  #model{flex:1; min-width:220px}
  select:disabled{opacity:.45; cursor:default}
  .speakers-row{display:flex; gap:12px; align-items:center; margin-top:12px; flex-wrap:wrap}
  .toggle{display:flex; align-items:center; gap:8px; font-size:13.5px; cursor:pointer; user-select:none}
  .toggle input{width:16px; height:16px; accent-color:var(--amber)}
  button{
    font:600 14px -apple-system, BlinkMacSystemFont, sans-serif;
    border:none; border-radius:8px; padding:11px 22px; cursor:pointer;
    background:var(--ink); color:#f2efe8;
  }
  button:hover{background:#000}
  button:disabled{opacity:.4; cursor:default}
  button.quiet{background:transparent; color:var(--ink); border:1px solid var(--line)}
  button.quiet:hover{background:var(--panel-2)}
  select:focus-visible, button:focus-visible, .drop:focus-visible, .toggle input:focus-visible{outline:2px solid var(--amber); outline-offset:2px}
  .hint{margin-top:12px; font-size:12px; color:var(--muted)}

  .counter{display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px}
  .timecode{font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:34px; letter-spacing:.04em; color:var(--ink)}
  .timecode span{color:var(--muted); font-size:20px}
  .pct{font-family:var(--mono); font-size:16px; color:var(--amber)}
  .bar{height:8px; border-radius:4px; background:var(--panel-2); border:1px solid var(--line); margin:16px 0 14px; overflow:hidden}
  .bar i{display:block; height:100%; width:0; background:linear-gradient(90deg, var(--amber), var(--amber-bright)); transition:width .5s ease}
  .bar.busy i{width:38%; opacity:.55}
  .phase{font-size:13px; color:var(--muted)}
  .tail{
    margin-top:12px; font-family:var(--mono); font-size:13px; color:#4c473e;
    border-left:3px solid var(--amber); padding:2px 0 2px 12px; min-height:20px;
  }

  .stats{font-family:var(--mono); font-size:12.5px; color:var(--muted); margin-bottom:14px}
  .warn{font-size:12.5px; color:var(--err); margin:-6px 0 12px}
  pre{
    background:var(--panel-2); border:1px solid var(--line); border-radius:8px;
    padding:16px; max-height:340px; overflow:auto; white-space:pre-wrap;
    font:12.5px/1.6 var(--mono); color:var(--ink);
  }
  .row{display:flex; gap:10px; flex-wrap:wrap; margin-top:16px}
  .saved{margin-top:14px; font-size:12.5px; color:var(--muted); word-break:break-all}
  .error{color:var(--err); font-family:var(--mono); font-size:13px; white-space:pre-wrap}
  .hidden{display:none}
  footer{margin-top:20px; text-align:center; font-size:11.5px; color:#736e64; font-family:var(--mono); letter-spacing:.06em}
</style>
</head>
<body>
<div class="unit">
  <header>
    <div class="brand">Meeting<em>Scribe</em></div>
    <div class="local"><span class="dot" id="dot"></span><span id="dotlabel">LOCAL · OFFLINE</span></div>
  </header>

  <section class="panel" id="setup">
    <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose an audio or video file">
      <h2 id="dropTitle">Drop a recording here</h2>
      <p id="dropSub">or click to choose — audio or video (m4a, mp3, wav, mp4…)</p>
      <p class="file hidden" id="fileName"></p>
    </div>
    <input type="file" id="file" accept="audio/*,video/*,.m4a,.mp3,.wav,.aac,.flac,.ogg,.mp4,.mov,.webm" hidden>
    <div class="controls">
      <select id="model" aria-label="Whisper model">__MODEL_OPTIONS__</select>
      <button id="start" disabled>Transcribe</button>
    </div>
    <div class="speakers-row">
      <label class="toggle"><input type="checkbox" id="diar" checked> Label speakers</label>
      <select id="spk" aria-label="Number of speakers">__SPEAKER_OPTIONS__</select>
    </div>
    <p class="hint">First use downloads models once (small ≈ 244 MB, speaker models ≈ 45 MB), then everything runs offline on this Mac. Set the speaker count if you know it — it beats auto-detect.</p>
  </section>

  <section class="panel hidden" id="working">
    <div class="counter">
      <div class="timecode"><span id="proc">0:00:00</span> <span>/ <span id="total">–:––:––</span></span></div>
      <div class="pct" id="pct"></div>
    </div>
    <div class="bar" id="bar"><i id="fill"></i></div>
    <div class="phase" id="phaseText">Uploading…</div>
    <div class="tail" id="tail"></div>
  </section>

  <section class="panel hidden" id="result">
    <div class="stats" id="stats"></div>
    <p class="warn hidden" id="warn"></p>
    <pre id="preview"></pre>
    <div class="row">
      <button id="dlMd">Download .md</button>
      <button id="dlTxt" class="quiet">.txt</button>
      <button id="dlSrt" class="quiet">.srt</button>
      <button id="revealBtn" class="quiet">Reveal in Finder</button>
      <button id="again" class="quiet">New transcription</button>
    </div>
    <p class="saved" id="saved"></p>
  </section>

  <section class="panel hidden" id="failed">
    <p class="error" id="errText"></p>
    <div class="row"><button id="retry" class="quiet">Start over</button></div>
  </section>

  <footer>NOTHING LEAVES THIS MACHINE</footer>
</div>

<script>
const $ = id => document.getElementById(id);
let file = null, jobId = null, poller = null;

const drop = $('drop'), input = $('file'), startBtn = $('start');

function pickFile(f){
  if(!f) return;
  file = f;
  $('fileName').textContent = f.name + ' · ' + (f.size/1048576).toFixed(1) + ' MB';
  $('fileName').classList.remove('hidden');
  $('dropTitle').textContent = 'Ready';
  $('dropSub').textContent = 'Drop a different file to replace it';
  startBtn.disabled = false;
}
drop.addEventListener('click', () => input.click());
drop.addEventListener('keydown', e => { if(e.key==='Enter'||e.key===' '){ e.preventDefault(); input.click(); }});
input.addEventListener('change', () => pickFile(input.files[0]));
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => pickFile(e.dataTransfer.files[0]));
$('diar').addEventListener('change', () => { $('spk').disabled = !$('diar').checked; });

function show(id){
  ['setup','working','result','failed'].forEach(s => $(s).classList.toggle('hidden', s !== id));
}

const PHASES = {
  'queued': 'Waiting for a previous job to finish…',
  'loading-model': 'Loading Whisper — first use downloads it (this can take a few minutes)…',
  'downloading-speaker-models': 'Downloading speaker models (one time)…',
  'decoding': 'Reading the recording…',
  'diarizing': 'Listening for who\\u2019s speaking…',
  'transcribing': 'Transcribing on this Mac…'
};
const BUSY = new Set(['queued','loading-model','decoding']);

startBtn.addEventListener('click', async () => {
  if(!file) return;
  show('working');
  $('dot').classList.add('rec'); $('dotlabel').textContent = 'RUNNING · LOCAL';
  $('phaseText').textContent = 'Uploading recording…';
  $('bar').classList.add('busy'); $('pct').textContent = ''; $('tail').textContent = '';
  $('proc').textContent = '0:00:00'; $('total').textContent = '–:––:––';

  const fd = new FormData();
  fd.append('audio', file);
  fd.append('model', $('model').value);
  fd.append('diarize', $('diar').checked ? '1' : '0');
  fd.append('speakers', $('spk').value);
  try{
    const r = await fetch('/transcribe', {method:'POST', body:fd});
    const j = await r.json();
    if(!r.ok) throw new Error(j.error || 'Upload failed');
    jobId = j.job_id;
    poller = setInterval(poll, 900);
  }catch(err){ fail(err.message); }
});

async function poll(){
  let j;
  try{
    const r = await fetch('/status/' + jobId);
    j = await r.json();
  }catch(e){ return; }

  if(j.phase === 'done'){
    clearInterval(poller);
    $('dot').classList.remove('rec'); $('dotlabel').textContent = 'LOCAL · OFFLINE';
    let s = j.total + ' · ' + j.language;
    if(j.speakers > 0) s += ' · ' + j.speakers + ' speakers';
    s += ' · ' + j.segments + ' segments';
    $('stats').textContent = s;
    $('warn').classList.toggle('hidden', !j.warning);
    if(j.warning) $('warn').textContent = j.warning;
    $('preview').textContent = j.markdown;
    $('saved').textContent = 'Saved to ' + j.outdir;
    show('result');
    return;
  }
  if(j.phase === 'error'){ clearInterval(poller); fail(j.error); return; }

  $('phaseText').textContent = PHASES[j.phase] || '…';
  $('bar').classList.toggle('busy', BUSY.has(j.phase));
  if(!BUSY.has(j.phase)){
    $('fill').style.width = (j.progress*100).toFixed(1) + '%';
    $('pct').textContent = Math.round(j.progress*100) + '%';
  } else { $('pct').textContent = ''; }
  if(j.phase === 'transcribing' || j.phase === 'diarizing'){
    $('proc').textContent = j.processed; $('total').textContent = j.total;
  }
  $('tail').textContent = j.tail || '';
}

function fail(msg){
  $('dot').classList.remove('rec'); $('dotlabel').textContent = 'LOCAL · OFFLINE';
  $('errText').textContent = 'Something went wrong.\\n' + msg;
  show('failed');
}

$('dlMd').addEventListener('click', () => location.href = '/download/'+jobId+'/md');
$('dlTxt').addEventListener('click', () => location.href = '/download/'+jobId+'/txt');
$('dlSrt').addEventListener('click', () => location.href = '/download/'+jobId+'/srt');
$('revealBtn').addEventListener('click', () => fetch('/reveal/'+jobId, {method:'POST'}));
function reset(){
  file = null; jobId = null; input.value = '';
  $('fileName').classList.add('hidden');
  $('dropTitle').textContent = 'Drop a recording here';
  $('dropSub').textContent = 'or click to choose — audio or video (m4a, mp3, wav, mp4…)';
  startBtn.disabled = true;
  show('setup');
}
$('again').addEventListener('click', reset);
$('retry').addEventListener('click', reset);
</script>
</body>
</html>
""".replace("__MODEL_OPTIONS__", MODEL_OPTIONS).replace("__SPEAKER_OPTIONS__", SPEAKER_OPTIONS)


# ------------------------------------------------------------------- main ---

def free_port():
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", PREFERRED_PORT))
        return PREFERRED_PORT
    except OSError:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  MeetingScribe is running at {url}")
    print("  Transcripts are saved to", OUTPUT_ROOT)
    print("  Keep this window open while you use it. Press Ctrl+C to quit.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
