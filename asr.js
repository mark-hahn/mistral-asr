#!/usr/bin/env node

// https://docs.mistral.ai/capabilities/audio/
// https://console.mistral.ai/usage

const DUMP_ALL_SEGS = false;

import fs from "fs";
import fsp from "fs/promises";
import path from "path";
import os from "os";
import {spawn} from "child_process";
import {fileURLToPath} from "url";
import {setTimeout as sleep} from "timers/promises";
import axios from "axios";
import FormData from "form-data";
import {Mistral} from "@mistralai/mistralai";
import util from "util";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const tmpDir     = path.join(__dirname, 'tmp');

/* ---------------- CLI argument parsing ---------------- */
const rawArgs = process.argv.slice(2);

const flagsKVP = new Map(
  rawArgs.filter(a => a.startsWith("--") && a.includes("="))
    .map(a => {const [k, v] = a.split("=", 2); return [k, v];})
);
const switches = new Set(rawArgs.filter(a => a.startsWith("--") && !a.includes("=")));
const positional = rawArgs.filter(a => !a.startsWith("--"));

function getNum(name, dflt) {
  if (flagsKVP.has(name)) {
    const n = Number(flagsKVP.get(name));
    if (Number.isFinite(n) && n >= 0) return n;
  }
  return dflt;
}
// 120s -> 5 Mb flac
const chunkSec =              getNum("--chunk-sec",  120);
const trimSec =               getNum("--trim-sec",    20);
const overlapSec =            getNum("--overlap-sec", 10);
const offsetSec =             chunkSec - trimSec - overlapSec - trimSec;
const minChunkSec =           trimSec + overlapSec + trimSec;
const audioQuality =          flagsKVP.get("--audio-quality") || "max";
const enablePreprocessing =  !switches.has("--no-preprocess");
const enableNoiseReduction = !switches.has("--no-denoise");
const testMins =              getNum("--test-mins", 0);
const apiTemperature =        getNum("--temperature", 0);
const apiResponseFormat =     flagsKVP.get("--response-format") || "verbose_json";
const apiPrompt =             flagsKVP.get("--prompt") || null;

// Audio quality settings
const AUDIO_CONFIGS = {
  low:    {rate: 16000, bitrate: "64k"},
  medium: {rate: 22050, bitrate: "128k"},
  high:   {rate: 44100, bitrate: "192k"},
  max:    {rate: 48000, bitrate: "256k"}
};
const audioConfig = AUDIO_CONFIGS[audioQuality];

/* ---------------- Input validation ---------------- */
if (positional.length === 0) {
  console.error("❌ Error: No input file specified");
  process.exit(1);
}
const inputPath = path.resolve(positional[0]);

/* ---------------- API Key and setup ---------------- */
const keyPath = path.resolve("secrets/mistral-asr-key.txt");
let apiKey;
try {
  apiKey = fs.readFileSync(keyPath, "utf8").trim();
} catch (e) {
  console.error(`❌ Unable to read API key from ${keyPath}: ${e.message}`);
  process.exit(1);
}

const model         = "voxtral-mini-latest";
const forceLanguage = "en";
const allowedExt    = new Set([".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]);
const fileLimit     = 19 * 1024 * 1024;

/* ---------------- format logging timestamp  (HH:MM:SS.t) ---------------- */
let scriptStart = Date.now();
function ts() {
  const secs    = Math.floor((Date.now() - scriptStart) / 1000);
  const hours   = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const seconds = secs % 60;
  return (
    String(hours)  .padStart(2, "0") + ":" +
    String(minutes).padStart(2, "0") + ":" +
    String(seconds).padStart(2, "0")
  );
}

/* ---------- format video timestamp (H:MM:SS.t) ---------- */
function vs(secs) {
  // work in deciseconds for clean rounding/carry
  let ds = Math.round(secs * 10);
  const neg = ds < 0;
  ds = Math.abs(ds);
  let hours   = Math.floor(ds / 36000); // 3600s * 10
  ds %= 36000;
  let minutes = Math.floor(ds / 600);   // 60s * 10
  ds %= 600;
  let seconds = Math.floor(ds / 10);
  let tenths  = ds % 10;
  if (hours > 9) { hours = 9; minutes = 59; seconds = 59; tenths = 9; }
  const out = '[' +
    String(hours) + ":" +
    String(minutes).padStart(2, "0") + ":" +
    String(seconds).padStart(2, "0") + "." +
    String(tenths) + ']';
  return neg ? "-" + out : out;
}

/* ---------------- Utility functions ---------------- */
async function pathExists(p) {
  try {
    await fsp.access(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function getSrtPath(videoPath) {
  const dir = path.dirname(videoPath);
  const baseName = path.basename(videoPath, path.extname(videoPath));
  return path.join(dir, `${baseName}.enx.srt`);
}

function isVideoFile(p) {
  return allowedExt.has(path.extname(p).toLowerCase());
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, {stdio: ["ignore", "pipe", "pipe"], ...opts});
    let out = "", err = "";
    p.stdout.on("data", d => out += d.toString());
    p.stderr.on("data", d => err += d.toString());
    p.on("error", reject);
    p.on("close", code => {
      if (code === 0) {
        resolve({out, err});
      } else {
        reject(new Error(`${cmd} exited ${code}\n${err || out}`));
      }
    });
  });
}

async function getDurationSec(file) {
  try {
    const {out} = await run("ffprobe", [
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=nw=1:nk=1",
      file
    ]);
    const sec = parseFloat(out.trim());
    return Number.isFinite(sec) ? Math.floor(sec) : 0;
  } catch (e) {
    console.warn(`Warning: Could not get duration for ${file}: ${e.message}`);
    return 0;
  }
}

/* ---------------- Audio processing ---------------- */
async function extractAudio(inputVideo, outWav) {
  const args = [
    "-y", "-i", inputVideo,
    "-ac", "1",
    "-ar", String(audioConfig.rate),
    "-b:a", audioConfig.bitrate,
    "-vn"
  ];
  if (testMins > 0) args.push("-t", String(testMins * 60));
  args.push(outWav);
  await run("ffmpeg", args);
}

/*
  filters that change energy floor, spectral content, and dynamic range 
  are what most ASR/VADs use to decide “speech vs. noise.”

   agate=threshold=0.001:ratio=2:attack=10:release=100"

1) agate (noise gate) — biggest lever on VAD
    Why: Directly alters the noise floor and tail of words; 
         too aggressive makes VAD think speech is silence.
    Sweep: threshold: 0.0005 → 0.001 → 0.003 → 0.01 (linear amp; 
           ≈ -66 → -60 → -50.5 → -40 dBFS)  ratio: 2 → 4 → 8
    attack/release: attack=5–20, release=80–300 ms 
          (short attack/release can chop syllables, confusing VAD)
    Baseline on/off test: run once without agate, once with your chosen settings.

0) Baseline:  ffmpeg -i in.wav -ac 1 -ar 16000 -c:a pcm_s16le out-baseline.wav

2) Bandlimit only: 
     ffmpeg -i in.wav -ac 1 -ar 16000 -af "highpass=f=80,lowpass=f=8000" 
            -c:a pcm_s16le out-bandlimit.wav

3) + Gentle compression:
     ffmpeg -i in.wav -ac 1 -ar 16000 
            -af "highpass=f=80,lowpass=f=8000,acompressor=threshold=0.003:ratio=3:attack=20:release=400" 
           -c:a pcm_s16le out-comp.wav

4) + Mild gate:
      ffmpeg -i in.wav -ac 1 -ar 16000 -
      af "highpass=f=80,lowpass=f=8000,acompressor=threshold=0.003:ratio=3:attack=20:release=400,agate=threshold=0.001:ratio=2:attack=10:release=120" 
      -c:a pcm_s16le out-gate.wav

6) Add RNNoise:
    ffmpeg -i in.wav -ac 1 -ar 16000 
    -af "highpass=f=80,lowpass=f=8000,acompressor=threshold=0.003:ratio=3:attack=20:release=400,arnndn=m=./arnndn-models/std.rnnn" 
    -c:a pcm_s16le out-denoise.wav

    # A) arnndn with standard model
    ffmpeg -i in.wav -ac 1 -ar 16000 -af "arnndn=m=./arnndn-models/std.rnnn" out-arnndn.wav
    
    # B) afftdn only
    ffmpeg -i in.wav -ac 1 -ar 16000 -af "afftdn=nf=-25" out-afftdn.wav
    
    # C) baseline (no denoise)
    ffmpeg -i in.wav -ac 1 -ar 16000 -c:a pcm_s16le out-baseline.wav
                        
*/

let haveDumpedFFmpeg = false;

async function preprocessAudio(inputWav, outputWav) {
  const filters = [];
  if (enableNoiseReduction) {
    filters.push(
      "highpass=f=80",
      "lowpass=f=8000",
      "acompressor=threshold=0.003:ratio=3:attack=30:release=1000",
      "agate=threshold=0.001:ratio=2:attack=10:release=100"
    );
  }
  filters.push(
    "equalizer=f=1000:width_type=h:width=500:g=2",
    "equalizer=f=3000:width_type=h:width=1000:g=1",
    "loudnorm=I=-16:TP=-1.5:LRA=11"
  );
  let audioFilter = filters.join(',');

  audioFilter = "highpass=f=80,lowpass=f=8000,dynaudnorm=f=150:g=3:m=3:s=8";

  if(!haveDumpedFFmpeg) console.log({audioFilter});
  haveDumpedFFmpeg = true;

  await run("ffmpeg", [
    "-y", "-i", inputWav,
    "-af", audioFilter,
    "-ac", "1",
    "-ar", String(audioConfig.rate),
    "-b:a", audioConfig.bitrate,
    "-f", "wav",
    outputWav
  ]);
}
// chunkSec=20 trimSec=3 overlapSec=3 offsetSec=11 minChunkSec=6
// t:trim o:overlap C:exclusive content
// 
// normal middle chunks ...
// tttoooCCCCCCCCooottt  tttoooCCCCCCCCooottt
//            tttoooCCCCCCCCooottt

// first chunk ...
// CCCCCCCCCCCCCCooottt  tttoooCCCCCCCCooottt
//            tttoooCCCCCCCCooottt

// min size last chunk ...
// tttoooCCCCCCCCooottt
//            tttoooCCC

async function getChunks(inWav) {
  const totalDuration = await getDurationSec(inWav);
  let chunkCount      = Math.ceil(totalDuration / offsetSec);
  const chunks        = [];
  for(let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
    const chunkStart = chunkIndex * offsetSec;
    const chunkEnd   = Math.min(chunkStart + chunkSec, totalDuration);
    if((chunkEnd - chunkStart) < minChunkSec) break;
    let trimStart    = chunkStart + trimSec;
    let trimEnd      = chunkEnd   - trimSec;
    let overlapStart = trimStart + overlapSec;
    let overlapEnd   = trimEnd   - overlapSec;
    if(chunkIndex == 0) {
      trimStart    = chunkStart;
      overlapStart = chunkStart;
    }
    if(chunkIndex == (chunkCount - 1)) {
      overlapEnd = chunkEnd;
      trimEnd    = chunkEnd;
    }
    const wavPath = 
        path.join(tmpDir, `chunk-${String(chunkIndex).padStart(3, "0")}.wav`);
    await run("ffmpeg", [
      "-y", "-i", inWav,
      "-ss", String(chunkStart.toFixed(2)),
      "-to", String(chunkEnd.toFixed(2)),
      "-c:a", "pcm_s16le",
      "-avoid_negative_ts", "make_zero",
      wavPath
    ]);
    chunks.push({ wavPath, chunkIndex, chunkStart, chunkEnd, 
                  trimStart, trimEnd, overlapStart, overlapEnd });
  }
  return chunks;
}

/* ---------------- Transcription ---------------- */
async function getFlac(wavPath) {
  const flacPath = path.join(tmpDir,
                   path.basename(wavPath, ".wav") + ".flac");
  await run("ffmpeg", [
    "-y", "-i", wavPath,
    "-c:a", "flac",
    flacPath
  ]);
  const statSize = (await fsp.stat(flacPath)).size;
  if (statSize > fileLimit) {
    console.error(`FLAC file too large: ${statSize} bytes > ${fileLimit} bytes`);
    process.exit(1);
  }
  return {
    path:     flacPath,
    mime:    "audio/flac",
    filename: path.basename(flacPath),
    size:     statSize
  };
}

const MAX_RETRIES   = 5;
const BASE_DELAY_MS = 5000;
const API_TIMEOUT   = 120000; // 2 mins

async function callApi(uploadInfo) {
  const buf = await fsp.readFile(uploadInfo.path);
  const apiStart = Date.now();
  let attempt = 0;
  while (true) {
    const form = new FormData();
    form.append("file", buf, {
      filename:    uploadInfo.filename,
      contentType: uploadInfo.mime
    });
    form.append("model", model);
    form.append("language", forceLanguage);
    form.append("timestamp_granularities", "segment");
    form.append("response_format", apiResponseFormat);
    form.append("temperature", String(apiTemperature));
    if (apiPrompt) form.append("prompt", apiPrompt);
      attempt ++;   
    let response = {}; 
    try {
      response = await axios.post(
        "https://api.mistral.ai/v1/audio/transcriptions", form, {
            headers: {Authorization: `Bearer ${apiKey}`, ...form.getHeaders()},
            timeout: API_TIMEOUT
        }
      );
    }
    catch(err) { 
      response.status = err.message; 
    }
    if (response.status === 200) {
      response.data.delay = Date.now() - apiStart;
      return response.data;
    }
    if(attempt > MAX_RETRIES) {
      console.error(`[${ts()}] API error: ${response.status
                                           }\n FATAL: max retries reached`);
      process.exit(1);
    }
    console.error(`[${ts()}] API error: ${response.status}, retrying`);
    if (attempt == 1) console.log('chunk, size:', uploadInfo.size, 
                                       '- file:', uploadInfo.filename);
    const delay = BASE_DELAY_MS * Math.pow(2, attempt - 1);
    await sleep(delay);
    continue;
  }
}
function processSegments(segments, chunkInfo) {
  if (!segments || segments.length === 0) return [];
  const processedSegments = [];
  console.log();
  for (const segment of segments) {
    if (segment.start === undefined || segment.end === undefined 
                                    || !segment.text?.trim()) {
      console.error(`Invalid segment (missing start/end/text), chunk ${
                         chunkInfo.chunkIndex }`);
      process.exit(1);
    }
    const start = chunkInfo.chunkStart + segment.start;
    const end   = chunkInfo.chunkStart + segment.end;

    console.log(`RAW: ${chunkInfo.chunkIndex}, ${
                     vs(start)}, ${vs(end)}, "${segment.text.trim()}"`);

    const processedSegment = {
      start, end,
      text:  segment.text.trim(),
      chunk: chunkInfo
    };
    if (start > chunkInfo.trimStart &&
        end   < chunkInfo.trimEnd) 
      processedSegments.push(processedSegment);
  }
  // console.log(JSON.stringify(processedSegments, null, 2));
  console.log();
  return processedSegments;
}

function toSrtTime(totalSec) {
  const totalMs = Math.max(0, Math.round(totalSec * 1000));
  const h = String(Math.floor(totalMs / 3600000)).padStart(2, "0");
  const m = String(Math.floor((totalMs % 3600000) / 60000)).padStart(2, "0");
  const s = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, "0");
  const ms3 = String(totalMs % 1000).padStart(3, "0");
  return `${h}:${m}:${s},${ms3}`;
}

/* ---------------- SRT generation ---------------- */
function writeSRT(segments, outputPath) {
  if (!segments || segments.length === 0) {
    console.error("Video has no segments to write:", outputPath);
    process.exit(1);
  }
  const sortedSegments = segments.sort((a, b) => a.start - b.start);

  let lastStart = +1e9;
  let lastEnd   = -1e9;
  let lastText  = null;
  let skipSeg;
  const segOut = [];
  for (const segment of sortedSegments) {
    const start = segment.start;
    const end   = segment.end;
    const text  = segment.text.trim();
    try {
      skipSeg  = true;

      if (text.length == 0) continue;
      if(text == lastText)  continue;
      
      if((start > lastStart && start < lastEnd) ||
         (end   > lastStart && end   < lastEnd)) {
        if(text === lastText) continue;
        console.log(`\n[${ts()}] Overlapping segments ...`);
        console.log(`A ${vs(lastStart)}, ${vs(lastEnd)}, "${lastText}"`); 
        console.log(`B ${vs(start)}, ${vs(end)}, "${text}"`);
        if(text.length > lastText.length) {
          console.log('Using A');
          continue;
        }
        console.log('Using B');
        segOut.pop();
      }
      skipSeg  = false;
    }
    finally {
      if(!skipSeg) segOut.push({ start, end, text });
      lastStart = start;
      lastEnd   = end;
      lastText  = text;
    }
  }
  let srtContent = "";
  let index = 0;
  for (const seg of segOut) {
    const startTime = toSrtTime(seg.start);
    const endTime   = toSrtTime(seg.end);
    srtContent += `${++index}\n`;
    srtContent += `${startTime} --> ${endTime}\n`;
    srtContent += `${seg.text}\n\n`;
  }
  fs.writeFileSync(outputPath, srtContent, "utf8");
  console.log(`\n[${ts()}] Wrote: ${path.basename(outputPath)}`);
}

/* ---------------- Main processing function ---------------- */
async function processOneVideo(videoPath) {
  console.log(`\n[${ts()}] Processing: ${path.basename(videoPath)}`);
  const videoName = path.basename(videoPath, path.extname(videoPath));
  const srtPath   = getSrtPath(videoPath);
  if (await pathExists(srtPath)) {
    console.log(`\n${videoName}: Enhanced SRT already exists, skipping.`);
    return;
  }
  const rawWavFile       = path.join(tmpDir, "audio_raw.wav");
  const processedWavFile = path.join(tmpDir, "audio_processed.wav");
  try {
    await extractAudio(videoPath, rawWavFile);
    let finalWavFile = rawWavFile;
    if (enablePreprocessing) {
      console.log(`[${ts()}] Preprocessing audio...`);
      await preprocessAudio(rawWavFile, processedWavFile);
      finalWavFile = processedWavFile;
    }
    const totalDur = await getDurationSec(finalWavFile);
    const chunks   = await getChunks(finalWavFile);
    console.log(`[${ts()}] Duration: ${totalDur.toFixed(0)}s, ${chunks.length} chunks`);
    const allSegments = [];
    for (const chunkInfo of chunks) {
      try {
        const uploadInfo = await getFlac(chunkInfo.wavPath);
        const apiData    = await callApi(uploadInfo);
        if (apiData.segments && apiData.segments.length > 0) {
          const processedSegments = processSegments(apiData.segments, chunkInfo);
          allSegments.push(...processedSegments);
          if(DUMP_ALL_SEGS)
            console.log(`[${ts()}] Chunk ${
              (chunkInfo.chunkIndex ).toString().padStart(3)}/${
              (chunks.length).toString().padStart(3)} ${
              (chunkInfo.chunkStart).toString().padStart(4)}s ${
              (chunkInfo.chunkEnd).toString().padStart(4)}s, Size: ${
              Math.round(uploadInfo.size / 1e6).toString().padStart(2)}Mb, ${
              processedSegments.length.toString().padStart(3)} segments, api:${
              Math.round(apiData.delay/1000).toString().padStart(3)}s`);
          continue;
        } 
        else {
          console.log(`[${ts()}] Chunk ${
            (chunkInfo.chunkIndex ).toString().padStart(3)}: ${
            (chunkInfo.chunkStart).toString().padStart(4)}s ${
            (chunkInfo.chunkEnd).toString().padStart(4)}s, Size: ${
            Math.round(uploadInfo.size / 1e6).toString().padStart(2)}Mb, ⚠️ no segments`);
          continue;
        }
      } catch (err) {
        console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.chunkIndex }/${chunks.length}: ${chunkInfo.chunkStart.toFixed(0)}s-${chunkInfo.chunkEnd.toFixed(0)}s ❌ ${err.message}`);
      }
    }
    if (allSegments.length === 0) {
      console.error("No transcription segments found");
      process.exit(1);
    }
    if(DUMP_ALL_SEGS) {
      let lastIdx       = -1;
      let lastInOverlap = false;
      for(const segment of allSegments) {
        const chunkInfo = segment.chunk;
        const inOverlap = segment.start < chunkInfo.overlapStart
                      || segment.end   > chunkInfo.overlapEnd;
        if (chunkInfo.chunkIndex != lastIdx) {
          console.log(`\nChunk ${(chunkInfo.chunkIndex )}: ${
            chunkInfo.chunkStart}  ${
            chunkInfo.trimStart}  ${
            chunkInfo.overlapStart}  ${
            chunkInfo.overlapEnd}  ${
            chunkInfo.trimEnd}  ${
            chunkInfo.chunkEnd}`);
          lastIdx = chunkInfo.chunkIndex;
        }
        if(inOverlap != lastInOverlap) console.log();
        lastInOverlap = inOverlap;
        console.log(`${vs(segment.start)} ${vs(segment.end)} ${segment.text}`);
      }
    }
    const outputPath = getSrtPath(videoPath);
    writeSRT(allSegments, outputPath);
  } catch (err) {
    console.error(`[${ts()}] ❌ Failed to process: ${path.basename(videoPath)
                                                     }, ${err.message}`);
    process.exit(1);
  }
}

/* ---------------- Main execution ---------------- */
async function main() {
  console.log(`\nConfiguration:`);
  console.log(`   Test Mode:        ${testMins > 0 ? `${testMins} minutes` : 'OFF'}`);
  console.log(`   Chunk Duration:   ${chunkSec}s`);
  console.log(`   Trim Duration:    ${trimSec}s`);
  console.log(`   Overlap Duration: ${overlapSec}s`);
  console.log(`   Audio Quality:    ${audioQuality} (${audioConfig.rate}Hz, ${audioConfig.bitrate})`);
  console.log(`   Preprocessing:    ${enablePreprocessing}`);
  console.log(`   Noise Reduction:  ${enableNoiseReduction}`);
  console.log(`   API Model:        ${model}`);
  console.log(`   API Language:     ${forceLanguage}`);
  console.log(`   API Temperature:  ${apiTemperature}`);
  console.log(`   API Response:     ${apiResponseFormat}`);
  console.log(`   API Prompt:       ${apiPrompt || 'None'}`);
  console.log(`   File Size Limit:  ${(fileLimit / 1024 / 1024).toFixed(1)}MB`);
  console.log();

  try {
    if (!(await pathExists(inputPath))) {
      throw new Error(`Input path does not exist: ${inputPath}`);
    }
    const stat = await fsp.stat(inputPath);
    const videoFiles = [];

    if (stat.isFile()) {
      if (!isVideoFile(inputPath)) {
        throw new Error(`File is not a supported video format: ${inputPath}`);
      }
      videoFiles.push(inputPath);
    } else if (stat.isDirectory()) {
      const files = await fsp.readdir(inputPath);
      for (const file of files) {
        const fullPath = path.join(inputPath, file);
        const fileStat = await fsp.stat(fullPath);
        if (fileStat.isFile() && isVideoFile(fullPath)) {
          videoFiles.push(fullPath);
        }
      }
    } else {
      throw new Error(`Input is neither file nor directory: ${inputPath}`);
    }
    if (videoFiles.length === 0) {
      throw new Error("No video files found");
    }
    console.log(`Found ${videoFiles.length} video file(s) to process`);
    let processed = 0;
    let failed = 0;
    for (const videoFile of videoFiles) {
      try {
        await processOneVideo(videoFile);
        processed++;
      } catch (err) {
        console.error(`[${ts()}] ❌ Failed: ${path.basename(videoFile)} - ${err.message}`);
        failed++;
      }
    }
    if (failed > 0) {
      process.exit(1);
    }
    console.log();
  } catch (err) {
    console.error(`[${ts()}] ❌ Fatal error: ${err.message}`);
    process.exit(1);
  } finally {
    if (typeof closeLogger === 'function') {
      closeLogger();
    }
  }
}

// Check dependencies and run
(async () => {
  try {
    await run("ffmpeg", ["-version"]);
    await run("ffprobe", ["-version"]);
    await main();
  } catch (err) {
    console.error(`[${ts()}] 💥 Error: ${err.message}`);
    process.exit(1);
  }
})();
