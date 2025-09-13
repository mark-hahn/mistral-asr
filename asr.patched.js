// --- asr runtime dirs (injected) ---
import os from "os";
import path from "path";
const SECRETS_DIR = process.env.ASR_SECRETS_DIR || path.resolve("secrets");
const TMP_BASE = process.env.ASR_TMPDIR || process.env.TMPDIR || process.env.TMP || process.env.TEMP || os.tmpdir();
// -----------------------------------

// Example usage in your code:
// const keyPath = path.join(SECRETS_DIR, "mistral-asr-key.txt");
// const tmpWav  = path.join(TMP_BASE, "audio_raw.wav");
