/* 

deepgram 
  api key: c8b001bf723ecd300628e1e382323ff9f5494230
  project: 1f01fb5c-c062-457b-a857-01fd68906ac3

  Speechmatics Rev AI Google Cloud

accuracy ...
  Speechmatics	    ~5–10%	$0.24/hr   recommended by chatgpt
  Google Cloud V2	  ~6–12%  $0.14/hour Dynamic Batch (Logged)

  Google Cloud V2 Dynamic Batch applies to Standard models: default, 
  command_and_search, latest_short, latest_long, phone_call, video, 
  and chirp (V2 only). If you’re doing subtitles, Dynamic Batch + latest_long 
  or chirp is fine when you don’t need fast turnaround; otherwise use normal batch.

  AWS Transcribe	  ~7–14%	$1.44/hour
  Azure Speech	    ~8–15%	$1.10/hour
  Gladia	          ~6–12%	???
  Deepgram Nova-2   ~5–12%  $0.25/hr BAD recommended by chatgpt
  Rev AI	          ~5–9%	  $0.20/hr BAD recommended by chatgpt


recommended by chatgpt ...
  Speechmatics (Batch) — very strong on accents/noise; word timings + diarization.
  Rev AI (Async + optional forced alignment) —
  Deepgram Nova-2 (Batch) — competitive accuracy + word timings + utterances.

API options to enable
  Word timestamps (must-have)
  Punctuation (let the model punctuate; you’ll keep punctuation attached to the last word’s end time)
  Diarization (optional for “speaker:” prefixes; turn off for plain subs)
  Language / domain hints if available
  Post-processing rules (what makes subs feel pro)
  Readability: 1–2 lines, ≤ 42 chars per line, target ≤ 17 CPS (chars/sec)
  Timing: 1.0–5.0 s per cue, add 100–200 ms lead-in & lead-out if room
  Gaps: start a new cue when next.start − prev.end ≥ 300 ms (tune 250–400 ms)
  Merge: keep merging words while under max duration & line length; break at punctuation if possible
  No overlaps: ensure next.start ≥ prev.end + 50–100 ms after padding
  Snap punctuation: attach comma/period to the last word and keep that word’s end time

*/

// wordsToSrt.js

function fmtTime(t) {
  const ms = Math.round(t * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const mm = ms % 1000;
  const pad = (n, w=2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(mm,3)}`;
}

/**
 * Convert word-level timestamps to SRT cues.
 * @param {Array<{word:string,start:number,end:number}>} words
 * @param {Object} [opt]
 */
function wordsToSrt(words, opt = {}) {
  const {
    maxCharsPerLine = 42,
    maxDuration = 5.0,      // seconds
    minDuration = 1.0,
    gapThreshold = 0.30,    // new cue if gap >= 300 ms
    leadPad = 0.12,         // 120 ms pad at start if room
    tailPad = 0.12,         // 120 ms pad at end if room
    maxCps = 17,            // chars per second
  } = opt;

  const cues = [];
  let buf = [];
  let cueStart = null;
  let lastEnd = null;

  const flush = () => {
    if (!buf.length) return;
    const text = buf.map(t => t.txt).join(" ").replace(/\s+([,.!?…])/g, "$1");
    const start = Math.max(0, cueStart - leadPad);
    const rawEnd = buf[buf.length - 1].end;
    let end = rawEnd + tailPad;
    // Enforce duration bounds
    let dur = end - start;
    if (dur < minDuration) end = start + minDuration;
    if (dur > maxDuration) end = start + maxDuration;
    // Ensure no overlap with previous cue
    const prev = cues[cues.length - 1];
    if (prev && start < prev.end + 0.05) {
      // bump start if needed
      const bump = (prev.end + 0.05) - start;
      cueStart += bump;
      end += bump;
    }
    // Re-wrap into 1–2 lines respecting maxCharsPerLine
    const lines = wrapText(text, maxCharsPerLine);
    cues.push({ start: cueStart - leadPad >= 0 ? cueStart - leadPad : cueStart, end, lines });
    buf = [];
    cueStart = null;
  };

  const wrapText = (t, limit) => {
    if (t.length <= limit) return [t];
    // simple 2-line balance by words
    const words = t.split(/\s+/);
    let line1 = "";
    for (const w of words) {
      const trial = line1 ? line1 + " " + w : w;
      if (trial.length <= limit) line1 = trial;
      else break;
    }
    const line2 = t.slice(line1.length).trim().replace(/^\s+/, "");
    if (!line1 || line2.length > limit) {
      // fallback: greedy reflow to ~half/half
      const mid = Math.floor(words.length / 2);
      return [words.slice(0, mid).join(" "), words.slice(mid).join(" ")];
    }
    return [line1, line2];
  };

  const cpsOK = (chars, dur) => (dur > 0 ? (chars / dur) <= maxCps : true);

  for (const w of words) {
    if (!cueStart) { cueStart = w.start; lastEnd = w.end; }
    const gap = w.start - lastEnd;
    const candidate = buf.concat([{ txt: w.word, end: w.end }]);
    const text = candidate.map(t => t.txt).join(" ").replace(/\s+([,.!?…])/g, "$1");
    const candDur = (w.end + tailPad) - (cueStart - leadPad);
    const tooLong = candDur > maxDuration;
    const tooFast = !cpsOK(text.length, Math.max(0.5, candDur));
    const forceBreak = gap >= gapThreshold || tooLong || tooFast;

    if (forceBreak && buf.length) flush();

    if (!buf.length) cueStart = w.start; // new cue
    buf.push({ txt: w.word, end: w.end });
    lastEnd = w.end;
  }
  flush();

  // Emit SRT
  return cues.map((c, i) => {
    const lines = c.lines.length === 1 ? c.lines
               : [c.lines[0], c.lines[1]];
    return [
      String(i + 1),
      `${fmtTime(c.start)} --> ${fmtTime(c.end)}`,
      ...lines,
      ""
    ].join("\n");
  }).join("\n");
}

// Example usage:
// const srt = wordsToSrt(wordsArray);
// fs.writeFileSync("out.srt", srt);

module.exports = { wordsToSrt };
