# MeetingScribe

Transcribe meeting recordings entirely on your own Mac, with speaker labels. Drop in an audio or video file and get a formatted meeting transcript — speaker turns (Speaker 1, Speaker 2, …), timestamps, clean plain text, and subtitles.

Transcription runs locally with OpenAI's Whisper model (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)); speaker diarization runs locally with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). No cloud, no API keys, no accounts.

<!-- Add a screenshot named screenshot.png to this folder and it will show up here -->
<!-- ![MeetingScribe](screenshot.png) -->

## Privacy — what talks to the internet

Nothing sends your audio anywhere. There are no APIs, accounts, or keys. The only network activity is a one-time download of open-source model weights: the Whisper model you pick (from Hugging Face's public CDN) and two small speaker-ID models (~45 MB, from GitHub). After that you can turn Wi-Fi off and it keeps working. Recordings, transcripts, and everything in between stay on your Mac.

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.9+ — check with `python3 --version`; if missing, run `xcode-select --install`

## Setup

1. Download this repository — click the green **Code** button above → **Download ZIP**, then unzip it (or `git clone` if you use git). Put the folder somewhere permanent like Applications or your home folder.
2. macOS blocks apps that aren't signed by an Apple-registered developer, so the first launch needs one approval. Right-click **Start MeetingScribe.command** → **Open** → **Open**. (Plain double-click is blocked the first time; after this one approval, double-click works forever.)
   - If macOS still refuses on a newer version: double-click it, click **Done** on the warning, then go to **System Settings → Privacy & Security**, scroll down, and click **Open Anyway**.
3. First launch installs its dependencies into a private `venv` folder (a few minutes) and opens the app in your browser at `http://127.0.0.1:8756`.

After the first time, just **double-click Start MeetingScribe.command** — a terminal window opens (that's the engine; leave it open while using the app) and your browser opens automatically. Close the terminal window or press Ctrl+C in it to quit. Tip: drag **Start MeetingScribe.command** into your Dock for one-click launching.

## Using it

Drag a recording onto the drop zone (m4a, mp3, wav, aac, flac, mp4, mov and most other formats — video files work, it uses the audio track), pick a model, and hit Transcribe. Keep the terminal window open while it runs.

**Speaker labels** are on by default. If you know how many people were in the meeting, set the speaker count — it's noticeably more accurate than auto-detect. Labels are "Speaker 1/2/3" in order of first appearance; find-and-replace with real names afterwards. Diarization works best with decent audio and distinct voices; heavy crosstalk or one laptop mic across a big conference room will blur it.

**Models:** small is the sweet spot for meetings; large-v3-turbo is noticeably better on crosstalk and jargon if you don't mind ~1.6 GB and slower runs; base for quick rough drafts.

**Big files:** file size doesn't matter much (a 1.5 GB recording is fine) — length is what costs time. Rough guide on Apple Silicon: the small model transcribes about 5–10× faster than realtime, so a 2-hour meeting takes roughly 15–25 minutes plus a few minutes for speaker ID. A progress counter shows exactly where it is.

Transcripts save automatically to `~/Documents/MeetingScribe/`, one folder per recording, in three formats: `transcript.md` (speaker turns + plain text), `transcript.txt`, and `transcript.srt` (subtitles, speaker-prefixed).

## How it works

1. The recording is decoded to 16 kHz mono audio once.
2. If speaker labels are on, sherpa-onnx runs pyannote segmentation + a TitaNet embedding model to cluster the audio into speakers.
3. faster-whisper transcribes the audio into timestamped segments.
4. Each transcript segment is matched to the speaker it overlaps most, segments are merged into speaker turns, and the result is written to Markdown, plain text, and SRT.

Everything runs on CPU by default for maximum compatibility.

## Project layout

```
MeetingScribe/
├── app.py                       # the whole application (Flask server + web UI)
├── Start MeetingScribe.command  # double-click to launch (installs on first run)
├── Install MeetingScribe.command# sets up the Python environment
├── README.md
├── LICENSE
└── .gitignore
```

The `venv/` (Python environment) and `models/` (downloaded weights) folders are created automatically on first run and are intentionally not tracked in git.

## Uninstalling

Delete this folder and `~/Documents/MeetingScribe/`. Whisper models cache in `~/.cache/huggingface/` if you want that space back.

## Troubleshooting

- **"python3: command not found"** → run `xcode-select --install` in Terminal, then relaunch.
- **Speaker labeling failed but the transcript still saved** → it fell back gracefully; usually a one-off download hiccup, try again.
- **Wrong speaker count in results** → set the exact count in the dropdown instead of auto.
- **Port already in use** → the app picks another free port automatically; the terminal window shows the URL.

## License

MIT — see [LICENSE](LICENSE). Built on faster-whisper (MIT), sherpa-onnx (Apache-2.0), and OpenAI Whisper (MIT).
