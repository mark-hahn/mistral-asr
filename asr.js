#!/usr/bin/env node

// https://docs.mistral.ai/capabilities/audio/
// https://console.mistral.ai/usage

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

// Configuration - ALL variables defined here
const CHUNK_SEC = getNum("--chunk-sec", 30);
const AUDIO_QUALITY = flagsKVP.get("--audio-quality") || "max";
const ENABLE_PREPROCESSING = !switches.has("--no-preprocess");
const ENABLE_NOISE_REDUCTION = !switches.has("--no-denoise");
const TEST_MINS = getNum("--test-mins", 0);
const API_TEMPERATURE = getNum("--temperature", 0);
const API_RESPONSE_FORMAT = flagsKVP.get("--response-format") || "verbose_json";
const API_PROMPT = flagsKVP.get("--prompt") || null;

const chunkSec     = CHUNK_SEC;
const halfChunkSec = Math.floor(chunkSec / 2);

// Audio quality settings
const AUDIO_CONFIGS = {
  low: {rate: 16000, bitrate: "64k"},
  medium: {rate: 22050, bitrate: "128k"},
  high: {rate: 44100, bitrate: "192k"},
  max: {rate: 48000, bitrate: "256k"}
};
const audioConfig = AUDIO_CONFIGS[AUDIO_QUALITY];

/* ---------------- Input validation ---------------- */
if (positional.length === 0) {
  console.error("❌ Error: No input file specified");
  process.exit(1);
}

const inputPath = path.resolve(positional[0]);
let logDir;

/* ---------------- API Key and setup ---------------- */
const KEY_PATH = path.resolve("secrets/mistral-asr-key.txt");
let API_KEY;

try {
  API_KEY = fs.readFileSync(KEY_PATH, "utf8").trim();
} catch (e) {
  console.error(`❌ Unable to read API key from ${KEY_PATH}: ${e.message}`);
  process.exit(1);
}

const MODEL = "voxtral-mini-latest";
const FORCE_LANGUAGE = "en";
const ALLOWED_EXT = new Set([".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]);
const FILE_LIMIT = 19 * 1024 * 1024;

let scriptStart = Date.now();

/* ---------------- Timestamped logging setup ---------------- */
function ts() {
  const d = new Date(Date.now() - scriptStart);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours()-16)}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
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
  return ALLOWED_EXT.has(path.extname(p).toLowerCase());
}

function toSrtTime(totalSec) {
  const totalMs = Math.max(0, Math.round(totalSec * 1000));
  const h = String(Math.floor(totalMs / 3600000)).padStart(2, "0");
  const m = String(Math.floor((totalMs % 3600000) / 60000)).padStart(2, "0");
  const s = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, "0");
  const ms3 = String(totalMs % 1000).padStart(3, "0");
  return `${h}:${m}:${s}`;
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
    return Number.isFinite(sec) ? sec : 0;
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
  if (TEST_MINS > 0) args.push("-t", String(TEST_MINS * 60));
  args.push(outWav);

  await run("ffmpeg", args);

  const stat = await fsp.stat(outWav);
}

async function preprocessAudio(inputWav, outputWav) {
  const filters = [];
  if (ENABLE_NOISE_REDUCTION) {
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
  const audioFilter = filters.join(',');
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

async function getChunks(inWav) {
  const totalDuration = await getDurationSec(inWav);
  const chunks = [];
  let chunkCount = Math.floor(totalDuration / halfChunkSec);
  for(let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
    const chunkStart = chunkIndex * halfChunkSec;
    const chunkEnd   = chunkStart + chunkSec;
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
    chunks.push({
      wavPath,
      index: chunkIndex,
      chunkStart, chunkEnd
    });
  }
  return chunks;
}

/* ---------------- Transcription ---------------- */
async function getFlacs(wavPath) {
  const flacPath = path.join(tmpDir,
                   path.basename(wavPath, ".wav") + ".flac");
  await run("ffmpeg", [
    "-y", "-i", wavPath,
    "-c:a", "flac",
    flacPath
  ]);
  const stat = await fsp.stat(flacPath);
  if (stat.size > FILE_LIMIT) {
    throw new Error(`FLAC file too large: ${stat.size} bytes > ${FILE_LIMIT} bytes`);
  }
  return {
    path: flacPath,
    mime: "audio/flac",
    filename: path.basename(flacPath)
  };
}

async function callApi(uploadInfo) {
  const buf = await fsp.readFile(uploadInfo.path);
  const form = new FormData();
  try {
      form.append("file", buf, {
      filename:    uploadInfo.filename,
      contentType: uploadInfo.mime
    });
    form.append("model", MODEL);
    form.append("language", FORCE_LANGUAGE);
    form.append("timestamp_granularities", "segment");
    form.append("response_format", API_RESPONSE_FORMAT);
    form.append("temperature", String(API_TEMPERATURE));
    if (API_PROMPT) form.append("prompt", API_PROMPT);

    const response = await axios.post(
      "https://api.mistral.ai/v1/audio/transcriptions",
      form,
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          ...form.getHeaders()
        },
        timeout: 60000
      }
    );
    return response.data;
  }
  catch (err) {
    console.error(`API call failed: ${err.message}`);
    return {};
  }
}

function processSegments(segments, chunkInfo) {
  if (!segments || segments.length === 0) return [];
  const processedSegments = [];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (seg.start === undefined || seg.end === undefined || !seg.text?.trim()) {
      console.error(`SKIPPED: Invalid segment (missing start/end/text)`);
      continue;
    }
    const processedSegment = {
      start:         chunkInfo.chunkStart + seg.start,
      end:           chunkInfo.chunkStart + seg.end,
      text:          seg.text.trim(),
      chunkIndex:    chunkInfo.index,
      chunkStart:    chunkInfo.chunkStart,
      chunkEnd:      chunkInfo.chunkEnd
    };
    processedSegments.push(processedSegment);
  }
  return processedSegments;
}

/* ---------------- SRT generation ---------------- */
function generateSRT(segments, outputPath) {
  if (!segments || segments.length === 0) {
    throw new Error("No segments to write");
  }

  let srtContent = "";
  let index = 1;

  const sortedSegments = segments.sort((a, b) => a.start - b.start);

  for (const seg of sortedSegments) {
    if (!seg.text || !seg.text.trim()) continue;

    const startTime = toSrtTime(seg.start);
    const endTime   = toSrtTime(seg.end);

    srtContent += `${index}\n`;
    srtContent += `${startTime} --> ${endTime}\n`;
    srtContent += `${seg.text.trim()}\n\n`;
    index++;
  }

  fs.writeFileSync(outputPath, srtContent, "utf8");
  console.log(`[${ts()}] ✓ ${path.basename(outputPath)} written (${index - 1} captions)`);
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
  const rawWavFile          = path.join(tmpDir, "audio_raw.wav");
  const processedWavFile    = path.join(tmpDir, "audio_processed.wav");
  const enablePreprocessing = ENABLE_PREPROCESSING;
  try {
    await extractAudio(videoPath, rawWavFile);
    let finalWavFile = rawWavFile;
    if (enablePreprocessing) {
      await preprocessAudio(rawWavFile, processedWavFile);
      finalWavFile = processedWavFile;
    }
    const totalDur = await getDurationSec(finalWavFile);
    const chunks   = await getChunks(finalWavFile);
    console.log(`[${ts()}] Duration: ${totalDur.toFixed(0)}s, ${chunks.length} chunks`);
    const allSegments = [];
    for (const chunkInfo of chunks) {
      try {
        const uploadInfo = await getFlacs(chunkInfo.wavPath, tmpDir);
        const response   = await callApi(uploadInfo);
        if (response.segments && response.segments.length > 0) {
          const processedSegments = processSegments(response.segments, chunkInfo);
          allSegments.push(...processedSegments);
          console.log(`[${ts()}] Chunk ${
            (chunkInfo.index + 1).toString().padStart(3)}: ${
            (chunkInfo.chunkStart).toString().padStart(4)} ${
            (chunkInfo.chunkEnd).toString().padStart(4)}${
            processedSegments.length.toString().padStart(3)} segments`);
          continue;
        } 
        else {
          console.log(`[${ts()}] Chunk ${
            (chunkInfo.index + 1).toString().padStart(3)}: ${
            (chunkInfo.chunkStart).toString().padStart(4)} ${
            (chunkInfo.chunkEnd).toString().padStart(4)} ⚠️ no segments`);
          continue;
        }
      } catch (err) {
        console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.index + 1}/${chunks.length}: ${chunkInfo.chunkStart.toFixed(0)}s-${chunkInfo.chunkEnd.toFixed(0)}s ❌ ${err.message}`);
      }
    }
    if (allSegments.length === 0) 
        throw new Error("No transcription segments found");
    let lastIdx      = -1;
    let lastInHalf2  = false;
    for(const segment of allSegments) {
      const inHalf2 = segment.start > segment.chunkStart + halfChunkSec;
      if (segment.chunkIndex != lastIdx) {
        console.log(`\nChunk ${(segment.chunkIndex + 1)}: ${
          segment.chunkStart}  ${
          segment.chunkEnd}`);
        lastIdx = segment.chunkIndex;
        lastInHalf2 = false;
      }
      else {
        if (inHalf2 != lastInHalf2) console.log();
        lastInHalf2 = inHalf2;
      }
      console.log(`${
         segment.start.toFixed(1).padStart(6)} ${
         segment.end.toFixed(1).padStart(6)}  ${
         segment.text}`);
    }
  } catch (err) {
    console.error(`[${ts()}] ❌ ${path.basename(videoPath)} | Failed to process: ${err.message}`);
    throw err;
  }
}

/* ---------------- Main execution ---------------- */
async function main() {
  console.log(`Voxtral Configuration:`);
  console.log(`   Audio Quality: ${AUDIO_QUALITY} (${audioConfig.rate}Hz, ${audioConfig.bitrate})`);
  console.log(`   Chunk Duration: ${CHUNK_SEC}s`);
  console.log(`   Preprocessing: ${ENABLE_PREPROCESSING}`);
  console.log(`   Noise Reduction: ${ENABLE_NOISE_REDUCTION}`);
  console.log(`   Test Mode: ${TEST_MINS > 0 ? `${TEST_MINS} minutes` : 'OFF'}`);
  console.log(`   API Model: ${MODEL}`);
  console.log(`   API Language: ${FORCE_LANGUAGE}`);
  console.log(`   API Temperature: ${API_TEMPERATURE}`);
  console.log(`   API Response Format: ${API_RESPONSE_FORMAT}`);
  console.log(`   API Prompt: ${API_PROMPT || 'None'}`);
  console.log(`   File Size Limit: ${(FILE_LIMIT / 1024 / 1024).toFixed(1)}MB`);
  console.log();

  try {
    if (!(await pathExists(inputPath))) {
      throw new Error(`Input path does not exist: ${inputPath}`);
    }
    const stat = await fsp.stat(inputPath);
    const videoFiles = [];
    logDir = path.dirname(inputPath);

    if (stat.isFile()) {
      if (!isVideoFile(inputPath)) {
        throw new Error(`File is not a supported video format: ${inputPath}`);
      }
      videoFiles.push(inputPath);
      logDir = path.dirname(inputPath);
    } else if (stat.isDirectory()) {
      const files = await fsp.readdir(inputPath);
      for (const file of files) {
        const fullPath = path.join(inputPath, file);
        const fileStat = await fsp.stat(fullPath);
        if (fileStat.isFile() && isVideoFile(fullPath)) {
          videoFiles.push(fullPath);
        }
      }
      logDir = inputPath;
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
