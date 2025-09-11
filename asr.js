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
// 300s -> 14 Mb flac
const chunkSec =              getNum("--chunk-sec",  300);
const trimSec =               getNum("--trim-sec",    20);
const overlapSec =            getNum("--overlap-sec", 20);
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

let scriptStart = Date.now();

/* ---------------- format timestamp for logging  ---------------- */
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
  if (testMins > 0) args.push("-t", String(testMins * 60));
  args.push(outWav);
  await run("ffmpeg", args);
}

async function preprocessAudio(inputWav, outputWav) {
  const filters = [];
  if (enableNoiseReduction) {
    filters.push(
      "highpass=f=80",
      "lowpass=f=8000",
      // acompressor=threshold=0.003 (0-1) means volume reduced to 1/ratio
      // threshold too low?
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
    const trimStart  = (chunkIndex == 0) ? chunkStart : chunkStart + trimSec;
    const trimEnd    = (chunkIndex == (chunkCount - 1)) 
                                         ? chunkEnd : chunkEnd - trimSec;
    if((chunkEnd - chunkStart) < minChunkSec) break;
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
                                       trimStart, trimEnd });
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

async function callApi(uploadInfo) {
  const buf = await fsp.readFile(uploadInfo.path);
  const form = new FormData();
  try {
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

    const response = await axios.post(
      "https://api.mistral.ai/v1/audio/transcriptions",
      form,
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
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
    const start = chunkInfo.chunkStart + seg.start;
    const end   = chunkInfo.chunkStart + seg.end;
    const processedSegment = {
      start, end,
      text:           seg.text.trim(),
      chunkIndex:     chunkInfo.chunkIndex,
      chunkStart:     chunkInfo.chunkStart,
      chunkEnd:       chunkInfo.chunkEnd,
    };
    if (start < chunkInfo.trimStart || 
        end   > chunkInfo.trimEnd) continue;
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
  console.log(`[${ts()}] ${path.basename(outputPath)} written (${index - 1} captions)`);
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
    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
      const chunkInfo = chunks[chunkIndex];
      try {
        const uploadInfo = await getFlac(chunkInfo.wavPath, chunkIndex);
        const response   = await callApi(uploadInfo);
        if (response.segments && response.segments.length > 0) {
          const processedSegments = processSegments(response.segments, chunkInfo);
          allSegments.push(...processedSegments);
          console.log(`[${ts()}] Chunk ${
            (chunkInfo.chunkIndex + 1).toString().padStart(3)}: ${
            (chunkInfo.chunkStart).toString().padStart(4)}s ${
            (chunkInfo.chunkEnd).toString().padStart(4)}s, Size: ${
            Math.round(uploadInfo.size / 1e6).toString().padStart(2)}Mb, ${
            processedSegments.length.toString().padStart(3)} segments`);
          continue;
        } 
        else {
          console.log(`[${ts()}] Chunk ${
            (chunkInfo.chunkIndex + 1).toString().padStart(3)}: ${
            (chunkInfo.chunkStart).toString().padStart(4)}s ${
            (chunkInfo.chunkEnd).toString().padStart(4)}s, Size: ${
            Math.round(uploadInfo.size / 1e6).toString().padStart(2)}Mb, ⚠️ no segments`);
          continue;
        }
      } catch (err) {
        console.log(`[${ts()}] ${path.basename(videoPath)} | Chunk ${chunkInfo.chunkIndex + 1}/${chunks.length}: ${chunkInfo.chunkStart.toFixed(0)}s-${chunkInfo.chunkEnd.toFixed(0)}s ❌ ${err.message}`);
      }
    }
    if (allSegments.length === 0) {
      console.error("No transcription segments found");
      process.exit(1);
    }
    let lastIdx       = -1;
    let lastInOverlap = false;
    for(const segment of allSegments) {
      const inOverlap   = segment.start < segment.chunkStart + segment
                       || segment.end  > segment.chunkStart + chunkSec - 5;
      if (segment.chunkIndex != lastIdx) {
        console.log(`\nChunk ${(segment.chunkIndex + 1)}: ${
          segment.chunkStart}  ${
          segment.chunkEnd}`);
        lastIdx = segment.chunkIndex;
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
  console.log(`   Audio Quality: ${audioQuality} (${audioConfig.rate}Hz, ${audioConfig.bitrate})`);
  console.log(`   Chunk Duration: ${chunkSec}s`);
  console.log(`   Preprocessing: ${enablePreprocessing}`);
  console.log(`   Noise Reduction: ${enableNoiseReduction}`);
  console.log(`   Test Mode: ${testMins > 0 ? `${testMins} minutes` : 'OFF'}`);
  console.log(`   API Model: ${model}`);
  console.log(`   API Language: ${forceLanguage}`);
  console.log(`   API Temperature: ${apiTemperature}`);
  console.log(`   API Response Format: ${apiResponseFormat}`);
  console.log(`   API Prompt: ${apiPrompt || 'None'}`);
  console.log(`   File Size Limit: ${(fileLimit / 1024 / 1024).toFixed(1)}MB`);
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
