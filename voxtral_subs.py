#!/usr/bin/env python3
"""
Usage: (old?)
  python voxtral_subs.py <input_media>
      [--device auto|cuda|cpu]
      [--chunk 10] [--stride 2]
      [--min-dur 0.8] [--max-dur 6.0] [--max-chars 84]

tweak if needed:  (old?)
  --chunk 8 --stride 2 --min-dur 0.7 --max-dur 5.0 --max-chars 74

voxtral_subs.py — Voxtral Mini (3B) subtitles with smoother timing,
cached under /mnt/card, auto-named output, and verbose per-chunk progress.

"""

import argparse, os, sys, math, tempfile, subprocess, shlex, time, regex as re
from pathlib import Path

# -------- cache & temp on /mnt/card -------- #
CACHE_ROOT = "/mnt/card/cache"
TMP_ROOT = "/mnt/card/tmp"
os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)
os.makedirs(os.path.join(CACHE_ROOT, "transformers"), exist_ok=True)
os.makedirs(os.path.join(CACHE_ROOT, "hub"), exist_ok=True)
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["TMPDIR"] = TMP_ROOT

def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        die("ffmpeg not found on PATH.")

def run_ffmpeg_extract_wav(input_path: str, wav_path: str, sr: int = 16000):
    cmd = ["ffmpeg","-y","-i",input_path,"-vn","-acodec","pcm_s16le","-ar",str(sr),"-ac","1",wav_path]
    print(">> extracting audio:", " ".join(shlex.quote(x) for x in cmd)); subprocess.run(cmd, check=True)

def s_to_srt_ts(t: float) -> str:
    if t is None or t < 0: t = 0.0
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-math.floor(t))*1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments, out_path: str):
    with open(out_path,"w",encoding="utf-8") as f:
        for i,seg in enumerate(segments,1):
            f.write(f"{i}\n{s_to_srt_ts(seg['start'])} --> {s_to_srt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

# -------- sentence splitting & timing helpers -------- #
import regex as re
_SENT_SPLIT = re.compile(r"""(?<=\S)(?:[.!?]+(?:["')\]]+)?)(?:\s+|$)""", re.X)

def split_sentences(text: str, max_chars: int = 84):
    text = re.sub(r"\s+"," ",text).strip()
    if not text: return []
    parts = re.split(_SENT_SPLIT, text)
    if len(parts) <= 1:
        parts = re.split(r'(?<=[.!?])\s+', text)
    out=[]
    for p in parts:
        p=p.strip()
        if not p: continue
        if len(p)<=max_chars: out.append(p); continue
        words=p.split(); buf=[]; cur=0
        for w in words:
            add=(1 if buf else 0)+len(w)
            if cur+add<=max_chars: buf.append(w); cur+=add
            else: out.append(" ".join(buf)); buf=[w]; cur=len(w)
        if buf: out.append(" ".join(buf))
    return out

def distribute_time(start_s: float, end_s: float, sentences):
    dur=max(0.2,end_s-start_s)
    if not sentences: return []
    lengths=[max(1,len(s)) for s in sentences]; total=float(sum(lengths))
    times=[dur*(l/total) for l in lengths]
    segs=[]; t=start_s
    for s,td in zip(sentences,times):
        segs.append({"start":float(t),"end":float(t+td),"text":s}); t+=td
    if segs: segs[-1]["end"]=float(end_s)
    return segs

def post_merge(segments, min_dur=0.8, max_dur=6.0, join_gap=0.25, max_chars=84):
    merged=[]
    for seg in segments:
        if not merged: merged.append(seg); continue
        prev=merged[-1]; gap=seg["start"]-prev["end"]; prev_dur=prev["end"]-prev["start"]
        if prev_dur<min_dur and gap<=join_gap:
            prev["end"]=seg["end"]; prev["text"]=(prev["text"]+" "+seg["text"]).strip()
        else:
            merged.append(seg)
    final=[]
    for seg in merged:
        dur=seg["end"]-seg["start"]
        if dur<=max_dur and len(seg["text"])<=max_chars: final.append(seg); continue
        text=seg["text"]; mid_t=seg["start"]+dur/2; mid_c=len(text)//2
        split_idx=None
        for i in range(max(1,mid_c-15), min(len(text)-1,mid_c+15)):
            if text[i] in [","," ","; ",":"]:
                split_idx=i; break
        if split_idx is None: split_idx=mid_c
        left=text[:split_idx].rstrip(", ").strip(); right=text[split_idx:].lstrip(", ").strip()
        if left:  final.append({"start":seg["start"],"end":mid_t,"text":left})
        if right: final.append({"start":mid_t,"end":seg["end"],"text":right})
    return final

# -------- Voxtral transcription (chunked) -------- #
def transcribe_voxtral(wav_path: str, model_id: str, device: str, chunk_s: int, stride_s: int,
                       min_dur: float, max_dur: float, max_chars: int, max_new_tokens: int, threads: int):
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    import soundfile as sf, numpy as np, torch, uuid

    # threads
    if threads and threads > 0:
        try: torch.set_num_threads(threads)
        except Exception: pass

    if device == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_arg = device

    print(f">> loading Voxtral model: {model_id} (device={device_arg})")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_ROOT)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=CACHE_ROOT,
        low_cpu_mem_usage=True,
    )
    model = model.to(device_arg)

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:  die(f"Unexpected sample rate {sr}; expected 16k.")

    samples_per_chunk = int(chunk_s * sr)
    stride_samples    = int(stride_s * sr)
    pos, n = 0, len(wav)

    raw_chunks=[]
    t0=time.time()
    import uuid
    tmp_chunks_dir = Path(TMP_ROOT) / f"voxtral_chunks_{uuid.uuid4().hex}"
    tmp_chunks_dir.mkdir(parents=True, exist_ok=True)

    try:
        idx=0
        total_chunks = (n + max(1, samples_per_chunk - stride_samples) - 1) // max(1, samples_per_chunk - stride_samples)
        while pos < n:
            seg = wav[pos:pos+samples_per_chunk]
            if len(seg)==0: break

            # write chunk wav
            chunk_path = tmp_chunks_dir / f"chunk_{idx:06d}.wav"
            sf.write(str(chunk_path), seg, sr)

            print(f">> [chunk {idx+1}] {pos/sr:8.2f}s → {min(n,pos+samples_per_chunk)/sr:8.2f}s : generating (max_new_tokens={max_new_tokens})")
            conversation = [{"role":"user","content":[{"type":"audio","path":str(chunk_path)}]}]
            inputs = processor.apply_chat_template(conversation)
            for k,v in list(inputs.items()):
                if hasattr(v,"to"): inputs[k] = v.to(device_arg)

            import torch
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

            prompt_len = inputs["input_ids"].shape[1]
            text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()

            start_s = pos / sr
            end_s   = min(n, pos + samples_per_chunk) / sr
            raw_chunks.append({"start":float(start_s),"end":float(end_s),"text":text})

            pos += max(1, samples_per_chunk - stride_samples)
            idx += 1
    finally:
        # cleanup
        try:
            for p in tmp_chunks_dir.glob("*.wav"):
                try: os.remove(p)
                except Exception: pass
            os.rmdir(tmp_chunks_dir)
        except Exception: pass

    print(f">> voxtral transcription done in {time.time()-t0:.1f}s; chunks={len(raw_chunks)}")

    # smooth timing
    fine=[]
    for seg in raw_chunks:
        sents = split_sentences(seg["text"], max_chars=max_chars)
        if not sents: continue
        fine.extend(distribute_time(seg["start"], seg["end"], sents))
    clean = post_merge(fine, min_dur=min_dur, max_dur=max_dur, max_chars=max_chars)
    return clean

# ------------------------------- Main ------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Voxtral Mini subtitles with smoother timing & progress.")
    p.add_argument("input", help="Input media file (.mp4, .mkv, .wav, etc.)")
    p.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507",
                   help="Voxtral model ID (default: Voxtral-Mini-3B-2507)")
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--chunk", type=int, default=10, help="Chunk length (seconds)")
    p.add_argument("--stride", type=int, default=2, help="Overlap (seconds)")
    p.add_argument("--min-dur", type=float, default=0.8)
    p.add_argument("--max-dur", type=float, default=6.0)
    p.add_argument("--max-chars", type=int, default=84)
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VOXTRAL_MAX_NEW_TOKENS", 256)),
                   help="Max tokens to generate per chunk (lower = faster)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("VOXTRAL_THREADS", "0")),
                   help="Torch CPU threads (0 = let PyTorch decide)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists(): die(f"Input not found: {args.input}")

    out_path = in_path.with_name(in_path.stem + "-enx.srt")

    check_ffmpeg()
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as td:
        wav_path = os.path.join(td, "audio16k.wav")
        run_ffmpeg_extract_wav(str(in_path), wav_path, sr=16000)
        segments = transcribe_voxtral(
            wav_path, args.model, args.device, args.chunk, args.stride,
            args.min_dur, args.max_dur, args.max_chars,
            args.max_new_tokens, args.threads
        )

    write_srt(segments, str(out_path))
    print(f">> subtitles written to {out_path}")

if __name__ == "__main__":
    main()
