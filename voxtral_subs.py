#!/usr/bin/env python3
"""
Create subtitles with Voxtral Small.

Usage:
  python voxtral_subs.py input.mp4 --out captions.srt
  # Optional WebVTT:
  python voxtral_subs.py input.mp4 --out captions.vtt --format vtt
"""

import argparse, os, sys, tempfile, math, subprocess, shlex, time
from pathlib import Path

# ---------- Utils ----------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        sys.exit("ffmpeg not found on PATH. Please install ffmpeg and try again.")

def run_ffmpeg_extract_wav(video_path: str, wav_path: str, sr: int = 16000):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        wav_path
    ]
    print(">> extracting audio:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

def s_to_srt_ts(t: float) -> str:
    if t is None or t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def s_to_vtt_ts(t: float) -> str:
    if t is None or t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def write_srt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()
            f.write(f"{i}\n{s_to_srt_ts(start)} --> {s_to_srt_ts(end)}\n{text}\n\n")

def write_vtt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()
            f.write(f"{s_to_vtt_ts(start)} --> {s_to_vtt_ts(end)}\n{text}\n\n")

def normalize_chunks(output) -> list:
    """
    Convert HF pipeline output into a list of {start, end, text}.
    Works with models that return 'chunks' or 'segments'.
    """
    segs = []
    # preferred: output["chunks"] with {"timestamp": (start, end), "text": ...}
    if isinstance(output, dict) and "chunks" in output and output["chunks"]:
        for ch in output["chunks"]:
            ts = ch.get("timestamp") or ch.get("timestamps")
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                segs.append({"start": float(ts[0] or 0.0), "end": float(ts[1] or 0.0), "text": ch.get("text", "")})
        return segs

    # fallback: a single dict with "text" only (no timestamps) — make one big segment
    if isinstance(output, dict) and "text" in output:
        segs.append({"start": 0.0, "end": None, "text": output["text"]})
        return segs

    # unknown structure
    raise ValueError("Unexpected transcription output structure; no 'chunks' found.")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate subtitles with Voxtral Small (Mistral ASR).")
    parser.add_argument("input", help="Input video or audio file (e.g., input.mp4)")
    parser.add_argument("--out", default="captions.srt", help="Output subtitle file (.srt or .vtt)")
    parser.add_argument("--format", choices=["srt", "vtt"], default=None, help="Force subtitle format")
    parser.add_argument("--model", default="mistralai/Voxtral-Small", help="HF model id")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--chunk", type=int, default=30, help="chunk length (seconds) for long files")
    parser.add_argument("--stride", type=int, default=5, help="overlap (seconds) between chunks")
    args = parser.parse_args()

    check_ffmpeg()

    # Decide output format
    fmt = args.format or (Path(args.out).suffix.lower().lstrip(".") or "srt")
    if fmt not in {"srt", "vtt"}:
        fmt = "srt"

    # Extract 16k mono wav (fast & robust for ASR)
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio16k.wav")
        run_ffmpeg_extract_wav(args.input, wav_path, sr=16000)

        # Load model
        print(">> loading model:", args.model)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model)

        device_map = None
        device_arg = 0 if args.device in ("auto", "cuda") else -1
        if args.device == "cpu":
            device_arg = -1

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device_arg,
            generate_kwargs={"return_timestamps": True},
            chunk_length_s=args.chunk,
            stride_length_s=(args.stride, args.stride),
        )

        print(">> transcribing …")
        t0 = time.time()
        out = pipe(wav_path, return_timestamps=True)
        dt = time.time() - t0
        print(f">> done in {dt:.1f}s")

        segments = normalize_chunks(out)

        # If any end is None (rare), patch with next start or +2s
        for i, seg in enumerate(segments):
            if seg["end"] is None:
                if i + 1 < len(segments):
                    seg["end"] = segments[i + 1]["start"]
                else:
                    seg["end"] = (seg["start"] or 0.0) + 2.0

        # Write subs
        out_path = Path(args.out).as_posix()
        if fmt == "vtt":
            write_vtt(segments, out_path)
        else:
            write_srt(segments, out_path)

        print(f">> subtitles written to: {out_path}")

        # Optional: print an ffmpeg mux command
        in_suffix = Path(args.input).suffix.lower()
        if in_suffix in {".mp4", ".mov", ".mkv"} and fmt == "srt":
            mux_cmd = f'ffmpeg -i {shlex.quote(args.input)} -i {shlex.quote(out_path)} -c copy -c:s mov_text output_with_subs.mp4'
            print(">> To embed subs into an MP4 (soft subtitles):")
            print("   " + mux_cmd)

if __name__ == "__main__":
    main()
