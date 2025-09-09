#!/usr/bin/env node

// https://docs.mistral.ai/capabilities/audio/
// https://console.mistral.ai/usage

/**
 * Usage:
 *   node voxtral-claude.js [options] <video-file-or-folder>
 * 
 * High accuracy Recommended:
 *   node voxtral-claude.js --audio-quality=high --chunk-sec=90 --silence-aware test
 * 
 * Maximum accuracy:
 *   node voxtral-claude.js --audio-quality=max --consensus --retries=3 --chunk-sec=60 test
 * 
 * Balanced Quality/Speed
 *   node voxtral-claude.js --audio-quality=medium --chunk-sec=120 test
 */

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

console.error("=== CORRECTED TIMESTAMP SCRIPT ===");
console.error("Raw arguments:", rawArgs);

const flagsKVP = new Map(
  rawArgs.filter(a => a.startsWith("--") && a.includes("="))
    .map(a => {const [k, v] = a.split("=", 2); return [k, v];})
);
const switches = new Set(rawArgs.filter(a => a.startsWith("--") && !a.includes("=")));
const positional = rawArgs.filter(a => !a.startsWith("--"));

function getNum(name, dflt) {
  if (flagsKVP.has(name)) {
    const n = Number(flagsKVP.get(name));
    if (Number.isFinite(n) && n > 0) return n;
  }
  return dflt;
}

// CORRECTED Configuration - simplified for debugging
const CHUNK_SEC = getNum("--chunk-sec", 90);
const SILENCE_AWARE = switches.has("--silence-aware");
const AUDIO_QUALITY = flagsKVP.get("--audio-quality") || "medium";
const ENABLE_PREPROCESSING = !switches.has("--no-preprocess");
const ENABLE_NOISE_REDUCTION = !switches.has("--no-denoise");

// Audio quality settings
const AUDIO_CONFIGS = {
  low: {rate: 16000, bitrate: "64k"},
  medium: {rate: 22050, bitrate: "128k"},
  high: {rate: 44100, bitrate: "192k"},
  max: {rate: 48000, bitrate: "256k"}
};
const audioConfig = AUDIO_CONFIGS[AUDIO_QUALITY];

console.error("CORRECTED Configuration:");
console.error(`  CHUNK_SEC: ${CHUNK_SEC}`);
console.error(`  SILENCE_AWARE: ${SILENCE_AWARE}`);
console.error(`  AUDIO_QUALITY: ${AUDIO_QUALITY} (${audioConfig.rate}Hz)`);

/* ---------------- Input validation ---------------- */
if (positional.length === 0) {
  console.error("❌ Error: No input file specified");
  console.error("Usage: node script.js [--chunk-sec=90] [--silence-aware] video.mkv");
  process.exit(1);
}

const inputPath = path.resolve(positional[0]);

/* ---------------- API Key and setup ---------------- */
const KEY_PATH = path.resolve("secrets/mistral-asr-key.txt");
let API_KEY;

try {
  API_KEY = fs.readFileSync(KEY_PATH, "utf8").trim();
  console.log(`✓ API key loaded`);
} catch (e) {
  console.error(`❌ Unable to read API key from ${KEY_PATH}: ${e.message}`);
  process.exit(1);
}

const MODEL = "voxtral-mini-latest";
const FORCE_LANGUAGE = "en";
const ALLOWED_EXT = new Set([".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]);
const FILE_LIMIT = 19 * 1024 * 1024;

/* ---------------- Timestamped logging setup ---------------- */
function ts() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${ms}`;
}

function setupLogger(logPath) {
  let logStream;

  try {
    logStream = fs.createWriteStream(logPath, {flags: "a"});
    logStream.write(`[${ts()}] === Voxtral CORRECTED logging started ===\n`);
    console.log(`📝 Logging to: ${logPath}`);
  } catch (e) {
    console.error(`⚠️  Warning: Could not setup log file ${logPath}: ${e.message}`);
    return () => { }; // Return no-op cleanup function
  }

  const originalLog = console.log;
  const originalError = console.error;
  const originalWarn = console.warn;

  const mirror = (level, args) => {
    const line = `[${ts()}] ${util.format(...args)}`;

    // Always write to console
    if (level === "error") {
      process.stderr.write(line + "\n");
    } else {
      process.stdout.write(line + "\n");
    }

    // Write to log file if available
    if (logStream && logStream.writable) {
      try {
        logStream.write(line + "\n");
      } catch (e) {
        console.log = originalLog;
        console.error = originalError;
        console.warn = originalWarn;
        console.warn(`⚠️  Log file write failed: ${e.message}`);
      }
    }
  };

  console.log = (...a) => mirror("log", a);
  console.error = (...a) => mirror("error", a);
  console.warn = (...a) => mirror("warn", a);

  return () => {
    try {
      if (logStream && logStream.writable) {
        logStream.write(`[${ts()}] === Voxtral CORRECTED logging ended ===\n`);
        logStream.end();
      }
      console.log = originalLog;
      console.error = originalError;
      console.warn = originalWarn;
    } catch (e) {
      // Ignore cleanup errors
    }
  };
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
  return path.join(dir, `${baseName}.enh.srt`);
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
    console.log(`Running: ${cmd} ${args.join(" ")}`);
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
  console.log(`Extracting audio at high quality...`);
  await run("ffmpeg", [
    "-y", "-i", inputVideo,
    "-ac", "1",
    "-ar", String(audioConfig.rate),
    "-b:a", audioConfig.bitrate,
    "-vn",
    outWav
  ]);
}

async function preprocessAudio(inputWav, outputWav) {
  console.log(`Preprocessing audio (quality: ${AUDIO_QUALITY})...`);

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

/* ---------------- CORRECTED: Simple, precise chunking with boundary fix ---------------- */
async function createPreciseChunks(inWav, outDir, chunkSec) {
  console.log(`Creating precise chunks (${chunkSec}s each, NO OVERLAP)...`);

  const totalDuration = await getDurationSec(inWav);
  console.log(`Total audio duration: ${totalDuration.toFixed(1)}s`);

  const chunks = [];
  let chunkStart = 0;
  let chunkIndex = 0;

  while (chunkStart < totalDuration) {
    const chunkEnd = Math.min(chunkStart + chunkSec, totalDuration);
    const chunkDuration = chunkEnd - chunkStart;

    // Skip very short final chunks
    if (chunkDuration < 5) {
      console.log(`Skipping final chunk of ${chunkDuration.toFixed(1)}s (too short)`);
      break;
    }

    const outputPath = path.join(outDir, `chunk-${String(chunkIndex).padStart(3, "0")}.wav`);

    // FIXED: Check if we're at the end and adjust extraction accordingly
    if (chunkEnd >= totalDuration) {
      // For the final chunk, extract from start to the very end
      console.log(`Final chunk: extracting from ${chunkStart.toFixed(1)}s to end (${totalDuration.toFixed(1)}s)`);
      await run("ffmpeg", [
        "-y", "-i", inWav,
        "-ss", String(chunkStart.toFixed(3)),
        "-to", String(totalDuration.toFixed(3)),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        outputPath
      ]);
    } else {
      // Normal chunk extraction with duration
      await run("ffmpeg", [
        "-y", "-i", inWav,
        "-ss", String(chunkStart.toFixed(3)),
        "-t", String(chunkDuration.toFixed(3)),
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
      duration: chunkDuration
    });

    console.log(`Chunk ${chunkIndex}: source ${chunkStart.toFixed(1)}s → ${chunkEnd.toFixed(1)}s (duration: ${chunkDuration.toFixed(1)}s)`);

    chunkStart = chunkEnd;
    chunkIndex++;
  }

  console.log(`Created ${chunks.length} chunks total`);
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
  console.log(`    FLAC size: ${(stat.size / 1024 / 1024).toFixed(2)} MB`);

  if (stat.size > FILE_LIMIT) {
    throw new Error(`FLAC file too large: ${stat.size} bytes > ${FILE_LIMIT} bytes`);
  }

  return {
    path: flacPath,
    mime: "audio/flac",
    filename: path.basename(flacPath)
  };
}

async function transcribeViaAxios(uploadPath, mime) {
  const buf = await fsp.readFile(uploadPath);
  const form = new FormData();

  form.append("file", buf, {
    filename: path.basename(uploadPath),
    contentType: mime
  });
  form.append("model", MODEL);
  form.append("language", FORCE_LANGUAGE);
  form.append("timestamp_granularities", "segment");
  form.append("response_format", "verbose_json");
  form.append("temperature", "0.1");

  const response = await axios.post(
    "https://api.mistral.ai/v1/audio/transcriptions",
    form,
    {
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        ...form.getHeaders()
      },
      timeout: 300000
    }
  );

  return response.data || {};
}

async function transcribeOneUpload(uploadInfo) {
  console.log(`  Transcribing ${uploadInfo.filename}...`);

  try {
    const resp = await transcribeViaAxios(uploadInfo.path, uploadInfo.mime);
    const segmentCount = Array.isArray(resp?.segments) ? resp.segments.length : 0;
    console.log(`  Found ${segmentCount} segments`);
    return resp;
  } catch (err) {
    console.error(`  Transcription failed: ${err.message}`);
    throw err;
  }
}

/* ---------------- CORRECTED: Timestamp processing without double-counting ---------------- */
function processSegments(segments, chunkInfo) {
  if (!segments || segments.length === 0) return [];

  console.log(`  Processing ${segments.length} segments for chunk ${chunkInfo.index}`);

  const processedSegments = [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];

    // Validate segment
    if (seg.start === undefined || seg.end === undefined || !seg.text?.trim()) {
      console.warn(`    Skipping invalid segment ${i}`);
      continue;
    }

    // CORRECTED LOGIC: API returns timestamps relative to chunk start (0 to chunk_duration)
    // So we simply add the chunk's start time in the source to get global timestamps
    const globalStart = chunkInfo.startTimeInSource + seg.start;
    const globalEnd = chunkInfo.startTimeInSource + seg.end;

    // Validate final timestamps
    if (globalEnd <= globalStart) {
      console.warn(`    Skipping segment ${i} with invalid duration: ${globalStart.toFixed(1)}s → ${globalEnd.toFixed(1)}s`);
      continue;
    }

    // Debug output for first few segments
    if (i < 2) {
      console.log(`    Segment ${i}: chunk_time=${seg.start.toFixed(1)}s → global_time=${globalStart.toFixed(1)}s (offset: +${chunkInfo.startTimeInSource.toFixed(1)}s)`);
    }

    processedSegments.push({
      start: globalStart,
      end: globalEnd,
      text: seg.text.trim(),
      chunkIndex: chunkInfo.index,
      chunkRelativeStart: seg.start,
      chunkRelativeEnd: seg.end
    });
  }

  console.log(`  Processed ${processedSegments.length}/${segments.length} valid segments`);
  return processedSegments;
}

/* ---------------- SRT generation ---------------- */
function generateSRT(segments, outputPath) {
  if (!segments || segments.length === 0) {
    throw new Error("No segments to write");
  }

  let srtContent = "";
  let index = 1;

  // Sort segments by start time to ensure proper order
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
  console.log(`✓ Enhanced SRT written: ${outputPath} (${index - 1} captions)`);
}

/* ---------------- Main processing function ---------------- */
async function processOneVideo(videoPath) {
  const videoName = path.basename(videoPath, path.extname(videoPath));
  const srtPath = getEnhancedSrtPath(videoPath);

  if (await pathExists(srtPath)) {
    console.log(`✓ ${videoName}: Enhanced SRT already exists, skipping.`);
    return;
  }

  console.log(`\n🎬 Processing: ${videoName}`);

  const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), "voxtral-"));
  const rawWavFile = path.join(tmpDir, "audio_raw.wav");
  const processedWavFile = path.join(tmpDir, "audio_processed.wav");

  try {
    // Extract audio
    await extractAudioHighQuality(videoPath, rawWavFile);

    // Preprocess if enabled
    let finalWavFile = rawWavFile;
    if (ENABLE_PREPROCESSING) {
      await preprocessAudio(rawWavFile, processedWavFile);
      finalWavFile = processedWavFile;
    }

    const totalDur = await getDurationSec(finalWavFile);
    console.log(`Duration: ${totalDur.toFixed(1)}s`);

    // Create precise chunks
    const chunkDir = path.join(tmpDir, "chunks");
    await fsp.mkdir(chunkDir);

    const chunks = await createPreciseChunks(finalWavFile, chunkDir, CHUNK_SEC);

    // Process each chunk
    const allSegments = [];

    for (const chunkInfo of chunks) {
      console.log(`Processing chunk ${chunkInfo.index + 1}/${chunks.length}: ${chunkInfo.startTimeInSource.toFixed(1)}s → ${chunkInfo.endTimeInSource.toFixed(1)}s`);

      try {
        // Prepare upload
        const uploadInfo = await prepareLosslessUpload(chunkInfo.path, tmpDir);

        // Transcribe
        const response = await transcribeOneUpload(uploadInfo);

        if (response.segments && response.segments.length > 0) {
          // CORRECTED: Process segments with simple offset addition
          const processedSegments = processSegments(response.segments, chunkInfo);
          allSegments.push(...processedSegments);

          console.log(`  ✓ Added ${processedSegments.length} segments to timeline`);
        } else {
          console.warn(`  ⚠️ No segments found for chunk ${chunkInfo.index + 1}`);
        }

      } catch (err) {
        console.error(`❌ Failed to process chunk ${chunkInfo.index + 1}: ${err.message}`);
      }
    }

    if (allSegments.length === 0) {
      throw new Error("No transcription segments found");
    }

    // Generate SRT
    generateSRT(allSegments, srtPath);

    // Show timing verification for debugging
    console.log(`📊 Timing verification:`);
    console.log(`  First segment: ${allSegments[0]?.start.toFixed(1)}s`);
    console.log(`  Last segment: ${allSegments[allSegments.length - 1]?.end.toFixed(1)}s`);
    console.log(`  Total video duration: ${totalDur.toFixed(1)}s`);

  } catch (err) {
    console.error(`❌ Failed to process ${videoName}: ${err.message}`);
    throw err;
  } finally {
    try {
      await fsp.rm(tmpDir, {recursive: true, force: true});
      console.log(`Cleaned up temp dir`);
    } catch (e) {
      console.warn(`Warning: Could not clean up ${tmpDir}: ${e.message}`);
    }
  }
}

/* ---------------- Main execution ---------------- */
async function main() {
  console.log(`[${ts()}] 🚀 Voxtral Enhanced - CORRECTED TIMESTAMP VERSION`);

  try {
    if (!(await pathExists(inputPath))) {
      throw new Error(`Input path does not exist: ${inputPath}`);
    }

    const stat = await fsp.stat(inputPath);
    const videoFiles = [];
    let logDir = path.dirname(inputPath);

    if (stat.isFile()) {
      if (!isVideoFile(inputPath)) {
        throw new Error(`File is not a supported video format: ${inputPath}`);
      }
      videoFiles.push(inputPath);
      logDir = path.dirname(inputPath);
    } else if (stat.isDirectory()) {
      console.log(`[${ts()}] 📂 Scanning directory for video files: ${inputPath}`);
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

    // Setup logging to the same directory as input
    const logPath = path.join(logDir, "voxtral-srt.log");
    const closeLogger = setupLogger(logPath);

    if (videoFiles.length === 0) {
      throw new Error("No video files found");
    }

    console.log(`[${ts()}] 📹 Found ${videoFiles.length} video file(s):`);
    videoFiles.forEach((file, i) => {
      const enhancedSrtPath = getEnhancedSrtPath(file);
      console.log(`[${ts()}]   ${i + 1}. ${path.basename(file)} → ${path.basename(enhancedSrtPath)}`);
    });

    // Process each video
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

    console.log(`[${ts()}] 📊 Summary:`);
    console.log(`[${ts()}]   ✅ Processed: ${processed}`);
    console.log(`[${ts()}]   ❌ Failed: ${failed}`);
    console.log(`[${ts()}]   📝 Log: ${logPath}`);

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
    // Check ffmpeg
    await run("ffmpeg", ["-version"]);
    await run("ffprobe", ["-version"]);
    console.log(`[${ts()}] ✓ FFmpeg found`);

    await main();
  } catch (err) {
    console.error(`[${ts()}] 💥 Error: ${err.message}`);
    process.exit(1);
  }
})();
