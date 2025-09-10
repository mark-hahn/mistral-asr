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
const __dirname = path.dirname(__filename);

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
    if (Number.isFinite(n) && n >= 0) return n; // Allow 0 for overlap
  }
  return dflt;
}

// Configuration - ALL variables defined here
const CHUNK_SEC = getNum("--chunk-sec", 30);
const CHUNK_OVERLAP = getNum("--chunk-overlap", 3);
const SILENCE_AWARE = switches.has("--silence-aware") || !switches.has("--no-silence-aware");
const AUDIO_QUALITY = flagsKVP.get("--audio-quality") || "max";
const ENABLE_PREPROCESSING = !switches.has("--no-preprocess");
const ENABLE_NOISE_REDUCTION = !switches.has("--no-denoise");
const TRANSCRIPTION_RETRIES = getNum("--retries", 3);
const ENABLE_CONSENSUS = switches.has("--consensus");
const TEST_MINS = getNum("--test-mins", 0);
const CONSENSUS_METHOD = flagsKVP.get("--consensus-method") || "simple";
const API_TEMPERATURE = getNum("--temperature", 0);
const API_RESPONSE_FORMAT = flagsKVP.get("--response-format") || "verbose_json";
const API_PROMPT = flagsKVP.get("--prompt") || null;
const ENABLE_DEBUG = switches.has("--debug");

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
  console.error("Usage: node script.js [--chunk-sec=90] [--silence-aware] video.mkv");
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
// const FILE_LIMIT = 19 * 1024 * 1024;
const FILE_LIMIT = 100 * 1024 * 1024;

/* ---------------- Timestamped logging setup ---------------- */
function ts() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function tsPrecise() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const pad3 = (n) => String(n).padStart(3, "0");
  return `${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

let debugLogPath = null;

function setupLogger(logDir) {
  if (ENABLE_DEBUG) {
    debugLogPath = path.join(logDir, "voxtral-debug.log");
    fs.writeFileSync(debugLogPath, `[${tsPrecise()}] === Voxtral Debug Session Started ===\n`, "utf8");
  }
  return () => { };
}

function debugLog(message) {
  if (ENABLE_DEBUG && debugLogPath) {
    fs.appendFileSync(debugLogPath, `[${tsPrecise()}] ${message}\n`, "utf8");
  }
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

function getEnhancedSrtPath(videoPath) {
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
  return `${h}:${m}:${s},${ms3}`;
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
async function extractAudioHighQuality(inputVideo, outWav) {
  const args = [
    "-y", "-i", inputVideo,
    "-ac", "1",
    "-ar", String(audioConfig.rate),
    "-b:a", audioConfig.bitrate,
    "-vn"
  ];

  if (TEST_MINS > 0) {
    args.push("-t", String(TEST_MINS * 60));
    debugLog(`Audio extraction: Duration limited to ${TEST_MINS * 60}s (${TEST_MINS} minutes)`);
  }

  args.push(outWav);

  debugLog(`Audio extraction command: ffmpeg ${args.join(' ')}`);
  debugLog(`Audio extraction: Input=${path.basename(inputVideo)}, Output=${path.basename(outWav)}`);
  debugLog(`Audio extraction: Quality=${audioConfig.rate}Hz, ${audioConfig.bitrate}`);

  await run("ffmpeg", args);

  const stat = await fsp.stat(outWav);
  debugLog(`Audio extraction: Output file size=${(stat.size / 1024 / 1024).toFixed(2)}MB`);
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

/* ---------------- Silence detection for smart chunking ---------------- */
async function detectSilenceBoundaries(wavFile, chunkSec, overlapSec = 0) {
  debugLog(`\n\nSilence detection: Starting analysis`);
  debugLog(`Silence detection: Input file=${path.basename(wavFile)}`);
  debugLog(`Silence detection: Target chunk size=${chunkSec}s, overlap=${overlapSec}s`);

  try {
    const {out, err} = await run("ffmpeg", [
      "-i", wavFile,
      "-af", "silencedetect=noise=-30dB:duration=0.5",
      "-f", "null",
      "-"
    ]);

    const silenceRegex = /silence_start: ([\d.]+)/g;
    const silenceTimes = [];
    let match;

    const output = out + err;
    while ((match = silenceRegex.exec(output)) !== null) {
      silenceTimes.push(parseFloat(match[1]));
    }

    debugLog(`Silence detection: Found ${silenceTimes.length} silence points: [${silenceTimes.map(t => t.toFixed(3)).join(', ')}]`);

    const totalDuration = await getDurationSec(wavFile);
    debugLog(`Silence detection: Total duration=${totalDuration.toFixed(6)}s`);

    const boundaries = [0];
    let currentPos = 0;
    let boundaryIndex = 0;

    while (currentPos < totalDuration) {
      const idealNext = currentPos + chunkSec;

      debugLog(`Silence detection: Boundary ${boundaryIndex} - currentPos=${currentPos.toFixed(6)}s, idealNext=${idealNext.toFixed(6)}s`);

      if (idealNext >= totalDuration) {
        if (totalDuration - currentPos > 5) {
          boundaries.push(totalDuration);
          debugLog(`Silence detection: Final boundary at ${totalDuration.toFixed(6)}s (duration=${(totalDuration - currentPos).toFixed(6)}s)`);
        } else {
          debugLog(`Silence detection: Skipping final short chunk (duration=${(totalDuration - currentPos).toFixed(6)}s < 5s)`);
        }
        break;
      }

      const searchStart = Math.max(idealNext - 10, currentPos + 30);
      const searchEnd = Math.min(idealNext + 10, totalDuration);

      debugLog(`Silence detection: Searching for silence between ${searchStart.toFixed(6)}s and ${searchEnd.toFixed(6)}s`);

      const candidateSilences = silenceTimes.filter(t => t >= searchStart && t <= searchEnd);

      let nextBoundary;
      if (candidateSilences.length > 0) {
        nextBoundary = candidateSilences.reduce((closest, current) =>
          Math.abs(current - idealNext) < Math.abs(closest - idealNext) ? current : closest
        );
        debugLog(`Silence detection: Found ${candidateSilences.length} candidate silences, chose ${nextBoundary.toFixed(6)}s (${Math.abs(nextBoundary - idealNext).toFixed(6)}s from ideal)`);
      } else {
        nextBoundary = idealNext;
        debugLog(`Silence detection: No silence found, using ideal position ${nextBoundary.toFixed(6)}s`);
      }

      boundaries.push(nextBoundary);
      currentPos = nextBoundary - overlapSec;
      debugLog(`Silence detection: Next currentPos=${currentPos.toFixed(6)}s (boundary - overlap)`);
      boundaryIndex++;
    }

    debugLog(`Silence detection: Final boundaries: [${boundaries.map(b => b.toFixed(6)).join(', ')}]`);
    return boundaries;

  } catch (error) {
    debugLog(`Silence detection: FAILED - ${error.message}`);
    console.warn(`[${ts()}] Warning: Silence detection failed, using fixed boundaries: ${error.message}`);

    const boundaries = [];
    const totalDuration = await getDurationSec(wavFile);
    let pos = 0;
    while (pos < totalDuration) {
      boundaries.push(pos);
      pos += chunkSec - overlapSec;
    }
    if (boundaries[boundaries.length - 1] < totalDuration) {
      boundaries.push(totalDuration);
    }

    debugLog(`Silence detection: Fallback boundaries: [${boundaries.map(b => b.toFixed(6)).join(', ')}]`);
    return boundaries;
  }
}

/* ---------------- Smart chunking with overlap and silence awareness ---------------- */
async function createPreciseChunks(inWav, outDir, chunkSec, overlapSec, silenceAware) {
  const totalDuration = await getDurationSec(inWav);
  const chunks = [];

  debugLog(`\n\nChunk creation: Starting chunking process`);
  debugLog(`Chunk creation: Input=${path.basename(inWav)}, duration=${totalDuration.toFixed(6)}s`);
  debugLog(`Chunk creation: chunkSec=${chunkSec}, overlapSec=${overlapSec}, silenceAware=${silenceAware}`);

  if (silenceAware && overlapSec > 0) {
    console.log(`[${ts()}] Using silence-aware chunking with ${overlapSec}s overlap`);
    debugLog(`Chunk creation: Mode=silence-aware+overlap`);

    const boundaries = await detectSilenceBoundaries(inWav, chunkSec, overlapSec);

    for (let i = 0; i < boundaries.length - 1; i++) {
      const chunkStart = Math.max(0, boundaries[i] - (i > 0 ? overlapSec : 0));
      const chunkEnd = boundaries[i + 1];
      const chunkDuration = chunkEnd - chunkStart;

      debugLog(`\n\nChunk ${i}: Silence-aware+overlap processing`);
      debugLog(`Chunk ${i}: boundary[${i}]=${boundaries[i].toFixed(6)}s, boundary[${i + 1}]=${boundaries[i + 1].toFixed(6)}s`);
      debugLog(`Chunk ${i}: chunkStart=${chunkStart.toFixed(6)}s (boundary - ${i > 0 ? overlapSec : 0}s overlap)`);
      debugLog(`Chunk ${i}: chunkEnd=${chunkEnd.toFixed(6)}s`);
      debugLog(`Chunk ${i}: chunkDuration=${chunkDuration.toFixed(6)}s`);

      if (chunkDuration < 5) {
        debugLog(`Chunk ${i}: SKIPPED - duration ${chunkDuration.toFixed(6)}s < 5s`);
        continue;
      }

      const outputPath = path.join(outDir, `chunk-${String(i).padStart(3, "0")}.wav`);

      debugLog(`Chunk ${i}: Extracting audio from ${chunkStart.toFixed(6)}s to ${chunkEnd.toFixed(6)}s`);
      debugLog(`Chunk ${i}: Output file=${path.basename(outputPath)}`);

      await run("ffmpeg", [
        "-y", "-i", inWav,
        "-ss", String(chunkStart.toFixed(6)),
        "-to", String(chunkEnd.toFixed(6)),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        outputPath
      ]);

      const stat = await fsp.stat(outputPath);
      debugLog(`Chunk ${i}: Output file size=${(stat.size / 1024 / 1024).toFixed(2)}MB`);

      chunks.push({
        path: outputPath,
        index: i,
        startTimeInSource: chunkStart,
        endTimeInSource: chunkEnd,
        actualStartInSource: boundaries[i],
        duration: chunkDuration,
        hasOverlap: i > 0 ? overlapSec : 0
      });

      debugLog(`Chunk ${i}: Created chunk object - actualStart=${boundaries[i].toFixed(6)}s, hasOverlap=${i > 0 ? overlapSec : 0}s`);
    }

  } else if (silenceAware) {
    console.log(`[${ts()}] Using silence-aware chunking`);
    debugLog(`Chunk creation: Mode=silence-aware only`);

    const boundaries = await detectSilenceBoundaries(inWav, chunkSec, 0);

    for (let i = 0; i < boundaries.length - 1; i++) {
      const chunkStart = boundaries[i];
      const chunkEnd = boundaries[i + 1];
      const chunkDuration = chunkEnd - chunkStart;

      debugLog(`\n\nChunk ${i}: Silence-aware processing`);
      debugLog(`Chunk ${i}: chunkStart=${chunkStart.toFixed(6)}s, chunkEnd=${chunkEnd.toFixed(6)}s, duration=${chunkDuration.toFixed(6)}s`);

      if (chunkDuration < 5) {
        debugLog(`Chunk ${i}: SKIPPED - duration ${chunkDuration.toFixed(6)}s < 5s`);
        continue;
      }

      const outputPath = path.join(outDir, `chunk-${String(i).padStart(3, "0")}.wav`);

      await run("ffmpeg", [
        "-y", "-i", inWav,
        "-ss", String(chunkStart.toFixed(6)),
        "-to", String(chunkEnd.toFixed(6)),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        outputPath
      ]);

      chunks.push({
        path: outputPath,
        index: i,
        startTimeInSource: chunkStart,
        endTimeInSource: chunkEnd,
        actualStartInSource: chunkStart,
        duration: chunkDuration,
        hasOverlap: 0
      });
    }

  } else if (overlapSec > 0) {
    console.log(`[${ts()}] Using fixed chunking with ${overlapSec}s overlap`);
    debugLog(`Chunk creation: Mode=fixed+overlap`);

    let chunkStart = 0;
    let chunkIndex = 0;

    while (chunkStart < totalDuration) {
      const chunkEnd = Math.min(chunkStart + chunkSec, totalDuration);
      const chunkDuration = chunkEnd - chunkStart;

      debugLog(`\n\nChunk ${chunkIndex}: Fixed+overlap processing`);
      debugLog(`Chunk ${chunkIndex}: chunkStart=${chunkStart.toFixed(6)}s, chunkEnd=${chunkEnd.toFixed(6)}s, duration=${chunkDuration.toFixed(6)}s`);

      if (chunkDuration < 5) {
        debugLog(`Chunk ${chunkIndex}: SKIPPED - duration ${chunkDuration.toFixed(6)}s < 5s`);
        break;
      }

      const outputPath = path.join(outDir, `chunk-${String(chunkIndex).padStart(3, "0")}.wav`);

      await run("ffmpeg", [
        "-y", "-i", inWav,
        "-ss", String(chunkStart.toFixed(6)),
        "-to", String(chunkEnd.toFixed(6)),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        outputPath
      ]);

      const actualStart = chunkIndex > 0 ? chunkStart + overlapSec : chunkStart;

      chunks.push({
        path: outputPath,
        index: chunkIndex,
        startTimeInSource: chunkStart,
        endTimeInSource: chunkEnd,
        actualStartInSource: actualStart,
        duration: chunkDuration,
        hasOverlap: chunkIndex > 0 ? overlapSec : 0
      });

      debugLog(`Chunk ${chunkIndex}: actualStart=${actualStart.toFixed(6)}s, hasOverlap=${chunkIndex > 0 ? overlapSec : 0}s`);

      chunkStart = chunkEnd - overlapSec;
      debugLog(`Chunk ${chunkIndex}: Next chunkStart=${chunkStart.toFixed(6)}s (current end - overlap)`);
      chunkIndex++;
    }

  } else {
    debugLog(`Chunk creation: Mode=fixed (no overlap)`);

    let chunkStart = 0;
    let chunkIndex = 0;

    while (chunkStart < totalDuration) {
      const chunkEnd = Math.min(chunkStart + chunkSec, totalDuration);
      const chunkDuration = chunkEnd - chunkStart;

      debugLog(`\n\nChunk ${chunkIndex}: Fixed processing`);
      debugLog(`Chunk ${chunkIndex}: chunkStart=${chunkStart.toFixed(6)}s, chunkEnd=${chunkEnd.toFixed(6)}s, duration=${chunkDuration.toFixed(6)}s`);

      if (chunkDuration < 5) {
        debugLog(`Chunk ${chunkIndex}: SKIPPED - duration ${chunkDuration.toFixed(6)}s < 5s`);
        break;
      }

      const outputPath = path.join(outDir, `chunk-${String(chunkIndex).padStart(3, "0")}.wav`);

      if (chunkEnd >= totalDuration) {
        await run("ffmpeg", [
          "-y", "-i", inWav,
          "-ss", String(chunkStart.toFixed(6)),
          "-to", String(totalDuration.toFixed(6)),
          "-c:a", "pcm_s16le",
          "-avoid_negative_ts", "make_zero",
          outputPath
        ]);
      } else {
        await run("ffmpeg", [
          "-y", "-i", inWav,
          "-ss", String(chunkStart.toFixed(6)),
          "-t", String(chunkDuration.toFixed(6)),
          "-c:a", "pcm_s16le",
          "-avoid_negative_ts", "make_zero",
          outputPath
        ]);
      }

      chunks.push({
        path: outputPath,
        index: chunkIndex,
        startTimeInSource: chunkStart,
        endTimeInSource: chunkEnd,
        actualStartInSource: chunkStart,
        duration: chunkDuration,
        hasOverlap: 0
      });

      chunkStart = chunkEnd;
      chunkIndex++;
    }
  }

  debugLog(`\n\nChunk creation: Created ${chunks.length} total chunks`);
  chunks.forEach((chunk, i) => {
    debugLog(`Final chunk ${i}: start=${chunk.startTimeInSource.toFixed(6)}s, end=${chunk.endTimeInSource.toFixed(6)}s, actual=${chunk.actualStartInSource.toFixed(6)}s, overlap=${chunk.hasOverlap}s`);
  });

  return chunks;
}

/* ---------------- Transcription ---------------- */
async function prepareLosslessUpload(partWavPath, tmpDir) {
  const flacPath = path.join(tmpDir, path.basename(partWavPath, ".wav") + ".flac");

  await run("ffmpeg", [
    "-y", "-i", partWavPath,
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

async function transcribeViaAxios(uploadPath, mime, chunkInfo) {
  debugLog(`\n\nAPI Call: Starting transcription for ${uploadPath}`);
  debugLog(`API Call: File size=${(await fsp.stat(uploadPath)).size} bytes`);
  debugLog(`API Call: MIME type=${mime}`);
  debugLog(`API Call: Chunk ${chunkInfo.index} (${chunkInfo.startTimeInSource.toFixed(6)}s - ${chunkInfo.endTimeInSource.toFixed(6)}s)`);

  const buf = await fsp.readFile(uploadPath);

  // const audioPath = logDir + '/' + path.basename(uploadPath);
  // debugLog(`writing flac file: ${audioPath}`);
  // await fsp.writeFile(audioPath, buf);

  const form = new FormData();

  form.append("file", buf, {
    filename: path.basename(uploadPath),
    contentType: mime
  });
  form.append("model", MODEL);
  form.append("language", FORCE_LANGUAGE);
  form.append("timestamp_granularities", "segment");
  form.append("response_format", API_RESPONSE_FORMAT);
  form.append("temperature", String(API_TEMPERATURE));

  if (API_PROMPT) {
    form.append("prompt", API_PROMPT);
  }

  debugLog(`API Call: Parameters sent to API:`);
  debugLog(`  model=${MODEL}`);
  debugLog(`  language=${FORCE_LANGUAGE}`);
  debugLog(`  timestamp_granularities=segment`);
  debugLog(`  response_format=${API_RESPONSE_FORMAT}`);
  debugLog(`  temperature=${API_TEMPERATURE}`);
  debugLog(`  prompt=${API_PROMPT || 'None'}`);
  debugLog(`  filename=${path.basename(uploadPath)}`);

  const startTime = Date.now();

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

  const endTime = Date.now();
  const duration = endTime - startTime;

  debugLog(`API Call: Response received in ${duration}ms`);
  debugLog(`API Call: HTTP status=${response.status}`);
  debugLog(`API Call: Response data keys=[${Object.keys(response.data || {}).join(', ')}]`);

  if (response.data && response.data.segments) {
    debugLog(`API Call: Segments count=${response.data.segments.length}`);
    debugLog(`API Call: Total text length=${response.data.text ? response.data.text.length : 0} chars`);
  }

  return response.data || {};
}

async function transcribeOneUpload(uploadInfo, chunkInfo) {
  debugLog(`\n\nTranscription attempt: Chunk ${chunkInfo.index}`);
  debugLog(`Transcription attempt: Upload file=${path.basename(uploadInfo.path)}`);
  debugLog(`Transcription attempt: MIME=${uploadInfo.mime}`);

  try {
    const resp = await transcribeViaAxios(uploadInfo.path, uploadInfo.mime, chunkInfo);
    const segmentCount = Array.isArray(resp?.segments) ? resp.segments.length : 0;

    debugLog(`Transcription attempt: SUCCESS - ${segmentCount} segments returned`);

    if (resp.segments && resp.segments.length > 0) {
      debugLog(`Transcription attempt: Segment details:`);
      resp.segments.forEach((seg, i) => {
        debugLog(` Segment ${i}: ${seg.start?.toFixed(6)}s - ${seg.end?.toFixed(6)}s: "${seg.text?.substring(0, 100)}${seg.text?.length > 100 ? '...' : ''}"`);
      });
    }

    return resp;
  } catch (err) {
    debugLog(`Transcription attempt: FAILED - ${err.message}`);
    throw err;
  }
}

/* ---------------- Retry and consensus logic ---------------- */
async function transcribeWithRetries(uploadInfo, maxRetries = 1, chunkInfo) {
  const attempts = [];
  let lastError;

  debugLog(`\n\nRetry logic: Starting transcription with up to ${maxRetries} attempts`);
  debugLog(`Retry logic: Chunk ${chunkInfo.index}, consensus=${ENABLE_CONSENSUS}`);

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    debugLog(`\nRetry logic: Attempt ${attempt}/${maxRetries}`);

    try {
      const result = await transcribeOneUpload(uploadInfo, chunkInfo);
      attempts.push(result);

      debugLog(`Retry logic: Attempt ${attempt} SUCCESS - ${result.segments?.length || 0} segments`);

      if (!ENABLE_CONSENSUS || attempt === 1) {
        debugLog(`Retry logic: Returning immediately (consensus=${ENABLE_CONSENSUS}, attempt=${attempt})`);
        return result;
      }

    } catch (error) {
      lastError = error;
      debugLog(`Retry logic: Attempt ${attempt} FAILED - ${error.message}`);

      if (attempt === maxRetries) {
        debugLog(`Retry logic: All attempts exhausted, throwing error`);
        throw lastError;
      }

      const delay = 1000 * attempt;
      debugLog(`Retry logic: Waiting ${delay}ms before retry`);
      await sleep(delay);
    }
  }

  if (attempts.length > 1 && ENABLE_CONSENSUS) {
    debugLog(`Retry logic: Using consensus with ${attempts.length} successful attempts`);

    if (CONSENSUS_METHOD === "simple") {
      debugLog(`Retry logic: Using simple consensus (most segments)`);
      const result = attempts.reduce((best, current) => {
        const bestCount = best.segments ? best.segments.length : 0;
        const currentCount = current.segments ? current.segments.length : 0;
        debugLog(`Retry logic: Simple consensus - comparing ${bestCount} vs ${currentCount} segments`);
        return currentCount > bestCount ? current : best;
      });
      debugLog(`Retry logic: Simple consensus selected result with ${result.segments?.length || 0} segments`);
      return result;
    } else {
      debugLog(`Retry logic: Using smart consensus`);
      return selectBestTranscription(attempts);
    }
  } else if (attempts.length > 0) {
    debugLog(`Retry logic: Returning first successful attempt (${attempts.length} total)`);
    return attempts[0];
  } else {
    debugLog(`Retry logic: No successful attempts, throwing error`);
    throw lastError || new Error("All transcription attempts failed");
  }
}

function selectBestTranscription(attempts) {
  if (attempts.length === 1) return attempts[0];

  debugLog(`\n\nSmart consensus: Evaluating ${attempts.length} transcription attempts`);

  const scored = attempts.map((attempt, index) => {
    const segments = attempt.segments || [];
    const segmentCount = segments.length;

    debugLog(`Smart consensus: Attempt ${index} - ${segmentCount} segments`);

    let totalTextLength = 0;
    let wordCount = 0;
    let avgConfidence = 0;

    for (const seg of segments) {
      if (seg.text) {
        totalTextLength += seg.text.length;
        wordCount += seg.text.trim().split(/\s+/).length;
      }
      if (seg.confidence !== undefined) {
        avgConfidence += seg.confidence;
      }
    }

    if (segmentCount > 0 && segments.some(s => s.confidence !== undefined)) {
      avgConfidence /= segmentCount;
    } else {
      avgConfidence = 1.0;
    }

    const score = {
      segmentCount: segmentCount,
      avgWordsPerSegment: segmentCount > 0 ? wordCount / segmentCount : 0,
      avgTextLength: segmentCount > 0 ? totalTextLength / segmentCount : 0,
      confidence: avgConfidence,
      totalWords: wordCount
    };

    debugLog(`Smart consensus: Attempt ${index} metrics:`);
    debugLog(`  segmentCount: ${score.segmentCount}`);
    debugLog(`  totalWords: ${score.totalWords}`);
    debugLog(`  avgWordsPerSegment: ${score.avgWordsPerSegment.toFixed(2)}`);
    debugLog(`  avgTextLength: ${score.avgTextLength.toFixed(2)}`);
    debugLog(`  confidence: ${score.confidence.toFixed(3)}`);

    const compositeScore =
      (score.segmentCount * 0.3) +
      (score.avgWordsPerSegment * 0.2) +
      (score.confidence * 0.3) +
      (Math.min(score.totalWords / 100, 1) * 0.2);

    debugLog(`Smart consensus: Attempt ${index} calculation:`);
    debugLog(`  (${score.segmentCount} * 0.3) + (${score.avgWordsPerSegment.toFixed(2)} * 0.2) + (${score.confidence.toFixed(3)} * 0.3) + (${Math.min(score.totalWords / 100, 1).toFixed(3)} * 0.2) = ${compositeScore.toFixed(4)}`);

    return {
      attempt,
      index,
      score,
      compositeScore
    };
  });

  scored.sort((a, b) => b.compositeScore - a.compositeScore);

  debugLog(`Smart consensus: Final ranking:`);
  scored.forEach((s, i) => {
    debugLog(`  ${i + 1}. Attempt ${s.index}: score=${s.compositeScore.toFixed(4)}, segments=${s.score.segmentCount}, words=${s.score.totalWords}`);
  });

  debugLog(`Smart consensus: Selected attempt ${scored[0].index} with score ${scored[0].compositeScore.toFixed(4)}`);

  if (process.env.DEBUG_CONSENSUS) {
    console.log(`[${ts()}] Consensus scoring:`);
    scored.forEach((s, i) => {
      console.log(`[${ts()}]   Attempt ${s.index}: score=${s.compositeScore.toFixed(2)}, segments=${s.score.segmentCount}, conf=${s.score.confidence.toFixed(2)}`);
    });
  }

  return scored[0].attempt;
}

/* ---------------- Overlap deduplication for segments ---------------- */
function deduplicateOverlappingSegments(allSegments) {
  if (!allSegments.length) return [];

  debugLog(`\n\nDeduplication: Starting with ${allSegments.length} segments`);

  allSegments.sort((a, b) => a.start - b.start);

  const deduplicated = [];
  let duplicatesFound = 0;

  for (let i = 0; i < allSegments.length; i++) {
    const segment = allSegments[i];
    let shouldInclude = true;
    let bestExistingMatch = null;
    let bestOverlapRatio = 0;
    let bestSimilarity = 0;

    // Check against all existing segments for overlaps
    for (const existing of deduplicated) {
      const overlapStart = Math.max(segment.start, existing.start);
      const overlapEnd = Math.min(segment.end, existing.end);
      const overlapDuration = Math.max(0, overlapEnd - overlapStart);

      const segmentDuration = segment.end - segment.start;
      const existingDuration = existing.end - existing.start;

      const overlapRatio1 = overlapDuration / segmentDuration;
      const overlapRatio2 = overlapDuration / existingDuration;
      const maxOverlapRatio = Math.max(overlapRatio1, overlapRatio2);

      // Only process significant overlaps (>70%)
      if (maxOverlapRatio > 0.7) {
        const similarity = calculateTextSimilarity(segment.text, existing.text);

        if (maxOverlapRatio > bestOverlapRatio) {
          bestExistingMatch = existing;
          bestOverlapRatio = maxOverlapRatio;
          bestSimilarity = similarity;
        }
      }
    }

    // Process the best match if found
    if (bestExistingMatch && bestOverlapRatio > 0.7) {
      debugLog(`\nDeduplication: Segment ${i} (${segment.start.toFixed(1)}s-${segment.end.toFixed(1)}s): "${segment.text.substring(0, 40)}..."`);
      debugLog(`  OVERLAP: ${(bestOverlapRatio * 100).toFixed(0)}% with existing segment at ${bestExistingMatch.start.toFixed(1)}s-${bestExistingMatch.end.toFixed(1)}s`);
      debugLog(`  TEXT SIMILARITY: ${(bestSimilarity * 100).toFixed(0)}%`);

      if (bestSimilarity > 0.8) {
        debugLog(`  EXCLUDED: Very similar text (>80%)`);
        shouldInclude = false;
        duplicatesFound++;
      } else if (bestSimilarity > 0.5) {
        if (segment.text.length > bestExistingMatch.text.length) {
          const existingIndex = deduplicated.indexOf(bestExistingMatch);
          deduplicated.splice(existingIndex, 1);
          debugLog(`  REPLACED: Existing segment (current is longer)`);
          shouldInclude = true;
        } else {
          debugLog(`  EXCLUDED: Existing segment is longer/better`);
          shouldInclude = false;
          duplicatesFound++;
        }
      } else {
        debugLog(`  INCLUDED: Low text similarity, both may be valid`);
        shouldInclude = true;
      }
    }

    if (shouldInclude) {
      deduplicated.push(segment);
    }
  }

  debugLog(`\nDeduplication: Result: ${allSegments.length} → ${deduplicated.length} segments (${duplicatesFound} duplicates removed)`);

  // Show sample of final segments
  if (deduplicated.length > 0) {
    debugLog(`\nDeduplication: Final segments sample:`);
    const sampleSize = Math.min(10, deduplicated.length);
    for (let i = 0; i < sampleSize; i++) {
      const seg = deduplicated[i];
      debugLog(`  ${seg.start.toFixed(1)}s: "${seg.text.substring(0, 50)}..."`);
    }
    if (deduplicated.length > sampleSize) {
      debugLog(`  ... and ${deduplicated.length - sampleSize} more segments`);
    }
  }

  return deduplicated;
}

function calculateTextSimilarity(text1, text2) {
  if (!text1 || !text2) return 0;

  const words1 = text1.toLowerCase().trim().split(/\s+/);
  const words2 = text2.toLowerCase().trim().split(/\s+/);

  if (words1.length === 0 && words2.length === 0) return 1;
  if (words1.length === 0 || words2.length === 0) return 0;

  const set1 = new Set(words1);
  const set2 = new Set(words2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  return intersection.size / union.size;
}

/* ---------------- Timestamp processing - NO overlap filtering ---------------- */
function processSegments(segments, chunkInfo) {
  if (!segments || segments.length === 0) return [];

  debugLog(`\nSegment processing: Chunk ${chunkInfo.index} has ${segments.length} raw segments`);
  debugLog(`Segment processing: Chunk timing - start=${chunkInfo.startTimeInSource.toFixed(6)}s, end=${chunkInfo.endTimeInSource.toFixed(6)}s`);
  debugLog(`Segment processing: Chunk overlap=${chunkInfo.hasOverlap}s, actualStart=${chunkInfo.actualStartInSource.toFixed(6)}s`);

  const processedSegments = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];

    debugLog(`\n Segment ${i}: Raw data`);
    debugLog(`  start=${seg.start?.toFixed(6)}s, end=${seg.end?.toFixed(6)}s`);
    debugLog(`  text="${seg.text?.substring(0, 100)}${seg.text?.length > 100 ? '...' : ''}"`);

    if (seg.start === undefined || seg.end === undefined || !seg.text?.trim()) {
      debugLog(`  SKIPPED: Invalid segment (missing start/end/text)`);
      continue;
    }

    // Calculate global timestamps  
    const globalStart = chunkInfo.startTimeInSource + seg.start;
    const globalEnd = chunkInfo.startTimeInSource + seg.end;

    debugLog(`  Global timestamps: ${globalStart.toFixed(1)}s - ${globalEnd.toFixed(1)}s`);

    if (globalEnd <= globalStart) {
      debugLog(`  SKIPPED: Invalid timing (end <= start)`);
      continue;
    }

    // NO OVERLAP FILTERING - let deduplication handle duplicates
    if (chunkInfo.hasOverlap > 0) {
      debugLog(`  OVERLAP INFO: Chunk has ${chunkInfo.hasOverlap}s overlap, keeping all segments for deduplication`);
    }

    const processedSegment = {
      start: globalStart,
      end: globalEnd,
      text: seg.text.trim(),
      chunkIndex: chunkInfo.index,
      chunkRelativeStart: seg.start,
      chunkRelativeEnd: seg.end,
      isInOverlap: chunkInfo.hasOverlap > 0 && globalStart < (chunkInfo.startTimeInSource + chunkInfo.hasOverlap)
    };

    debugLog(`  PROCESSED: Global=${globalStart.toFixed(6)}s-${globalEnd.toFixed(6)}s, inOverlap=${processedSegment.isInOverlap}`);

    processedSegments.push(processedSegment);
  }

  debugLog(`\nSegment processing: Chunk ${chunkInfo.index} result: ${segments.length} raw → ${processedSegments.length} processed segments`);
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
    const endTime = toSrtTime(seg.end);

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
  const videoName = path.basename(videoPath, path.extname(videoPath));
  const srtPath = getEnhancedSrtPath(videoPath);

  if (await pathExists(srtPath)) {
    console.log(`[${ts()}] ✓ ${videoName}: Enhanced SRT already exists, skipping.`);
    return;
  }

  console.log(`[${ts()}] 🎬 Processing: ${path.basename(videoPath)}`);

  const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), "voxtral-"));
  const rawWavFile = path.join(tmpDir, "audio_raw.wav");
  const processedWavFile = path.join(tmpDir, "audio_processed.wav");

  const chunkSec = CHUNK_SEC;
  const chunkOverlap = CHUNK_OVERLAP;
  const silenceAware = SILENCE_AWARE;
  const transcriptionRetries = TRANSCRIPTION_RETRIES;
  const enablePreprocessing = ENABLE_PREPROCESSING;

  try {
    await extractAudioHighQuality(videoPath, rawWavFile);

    let finalWavFile = rawWavFile;
    if (enablePreprocessing) {
      await preprocessAudio(rawWavFile, processedWavFile);
      finalWavFile = processedWavFile;
    }

    const totalDur = await getDurationSec(finalWavFile);

    const chunkDir = path.join(tmpDir, "chunks");
    await fsp.mkdir(chunkDir);

    const chunks = await createPreciseChunks(finalWavFile, chunkDir, chunkSec, chunkOverlap, silenceAware);

    console.log(`[${ts()}] Duration: ${totalDur.toFixed(0)}s, ${chunks.length} chunks`);

    const allSegments = [];

    for (const chunkInfo of chunks) {
      try {
        const uploadInfo = await prepareLosslessUpload(chunkInfo.path, tmpDir);

        const response = await transcribeWithRetries(uploadInfo, transcriptionRetries, chunkInfo);

        if (response.segments && response.segments.length > 0) {
          const processedSegments = processSegments(response.segments, chunkInfo);
          allSegments.push(...processedSegments);

          console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.index + 1}/${chunks.length}: ${chunkInfo.startTimeInSource.toFixed(0)}s-${chunkInfo.endTimeInSource.toFixed(0)}s ✓ ${processedSegments.length} segments`);
        } else {
          console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.index + 1}/${chunks.length}: ${chunkInfo.startTimeInSource.toFixed(0)}s-${chunkInfo.endTimeInSource.toFixed(0)}s ⚠️ no segments`);
        }

      } catch (err) {
        console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.index + 1}/${chunks.length}: ${chunkInfo.startTimeInSource.toFixed(0)}s-${chunkInfo.endTimeInSource.toFixed(0)}s ❌ ${err.message}`);
      }
    }

    if (allSegments.length === 0) {
      throw new Error("No transcription segments found");
    }

    const hasOverlap = chunkOverlap > 0;
    if (hasOverlap) {
      const originalCount = allSegments.length;
      const deduplicatedSegments = deduplicateOverlappingSegments(allSegments);
      console.log(`[${ts()}] ${path.basename(videoPath)} | Deduplication: ${originalCount} → ${deduplicatedSegments.length} segments`);
      allSegments.length = 0;
      allSegments.push(...deduplicatedSegments);
    }

    generateSRT(allSegments, srtPath);

    console.log(`[${ts()}] 📊 ${path.basename(videoPath)} | Timing verification:`);
    console.log(`[${ts()}]   First segment: ${allSegments[0]?.start.toFixed(0)}s`);
    console.log(`[${ts()}]   Last segment: ${allSegments[allSegments.length - 1]?.end.toFixed(0)}s`);
    console.log(`[${ts()}]   Total video duration: ${totalDur.toFixed(0)}s`);

  } catch (err) {
    console.error(`[${ts()}] ❌ ${path.basename(videoPath)} | Failed to process: ${err.message}`);
    throw err;
  } finally {
    try {
      await fsp.rm(tmpDir, {recursive: true, force: true});
      console.log(`[${ts()}] ${path.basename(videoPath)} | Cleaned up temp dir`);
    } catch (e) {
      console.warn(`[${ts()}] ${path.basename(videoPath)} | Warning: Could not clean up ${tmpDir}: ${e.message}`);
    }
  }
}

/* ---------------- Main execution ---------------- */
async function main() {
  console.log(`[${ts()}] 🚀 Voxtral Enhanced Configuration:`);
  console.log(`[${ts()}]   Audio Quality: ${AUDIO_QUALITY} (${audioConfig.rate}Hz, ${audioConfig.bitrate})`);
  console.log(`[${ts()}]   Chunk Duration: ${CHUNK_SEC}s`);
  console.log(`[${ts()}]   Chunk Overlap: ${CHUNK_OVERLAP}s`);
  console.log(`[${ts()}]   Silence Aware: ${SILENCE_AWARE}`);
  console.log(`[${ts()}]   Preprocessing: ${ENABLE_PREPROCESSING}`);
  console.log(`[${ts()}]   Noise Reduction: ${ENABLE_NOISE_REDUCTION}`);
  console.log(`[${ts()}]   Transcription Retries: ${TRANSCRIPTION_RETRIES}`);
  console.log(`[${ts()}]   Consensus Mode: ${ENABLE_CONSENSUS}`);
  if (ENABLE_CONSENSUS) {
    console.log(`[${ts()}]   Consensus Method: ${CONSENSUS_METHOD}`);
  }
  console.log(`[${ts()}]   Test Mode: ${TEST_MINS > 0 ? `${TEST_MINS} minutes` : 'OFF'}`);
  console.log(`[${ts()}]   Debug Mode: ${ENABLE_DEBUG}`);
  console.log(`[${ts()}]   API Model: ${MODEL}`);
  console.log(`[${ts()}]   API Language: ${FORCE_LANGUAGE}`);
  console.log(`[${ts()}]   API Temperature: ${API_TEMPERATURE}`);
  console.log(`[${ts()}]   API Response Format: ${API_RESPONSE_FORMAT}`);
  console.log(`[${ts()}]   API Prompt: ${API_PROMPT || 'None'}`);
  console.log(`[${ts()}]   File Size Limit: ${(FILE_LIMIT / 1024 / 1024).toFixed(1)}MB`);
  console.log(`[${ts()}] ===`);

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

    const logPath = path.join(logDir, "voxtral-srt.log");
    const closeLogger = setupLogger(logDir);

    if (videoFiles.length === 0) {
      throw new Error("No video files found");
    }

    console.log(`[${ts()}] 📹 Found ${videoFiles.length} video file(s) to process`);

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

    console.log(`[${ts()}] 📊 Summary: ✅ ${processed} processed, ❌ ${failed} failed`);

    if (failed > 0) {
      process.exit(1);
    }

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
