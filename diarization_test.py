"""
============================================================
  DIARIZATION TEST RUNNER
============================================================

USAGE
-----
1. Place audio files (.mp3 / .wav) in INPUT_DIR below.
2. Run:
       python diarization_test.py

3. Find JSON output per file in OUTPUT_DIR.

CONFIGURATION
-------------
Edit INPUT_DIR and OUTPUT_DIR below, or override via environment variables:
  INPUT_DIR   - folder that holds your input audio files
  OUTPUT_DIR  - folder where JSON results will be saved
  OPENAI_API_KEY - must be set in .env or your environment
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (looks for .env in the same directory as this script)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# CONFIGURE THESE PATHS
# ---------------------------------------------------------------------------
INPUT_DIR  = os.path.join(os.path.dirname(__file__), "test_input")   # <-- put audio files here
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output_second")  # <-- JSON results go here

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Force unbuffered output so logs appear immediately on Windows
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
handler.flush = sys.stdout.flush  # ensure each record flushes

logger = logging.getLogger("diarization_test")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Import standalone processor
# ---------------------------------------------------------------------------
from diarization_standalone import DiarizationProcessor


def process_audio_file(audio_path: str) -> dict:
    """Run the full diarization pipeline on a single audio file."""
    logger.info("=" * 60)
    logger.info(f"Processing: {audio_path}")
    logger.info("=" * 60)

    start = time.time()
    processor = DiarizationProcessor(audio_path=audio_path, logger=logger)
    
    # Dummy metadata to satisfy production prompts
    metadata = {
        "brand": "Dish",
        "call_language": "HINDI",
        "ucid": Path(audio_path).stem,
        "call_id": "LOCAL_TEST_" + Path(audio_path).stem
    }
    
    result = processor.process(metadata=metadata)
    elapsed   = time.time() - start

    result["processing_time_seconds"] = round(elapsed, 2)
    logger.info(f"Completed in {elapsed:.2f}s")
    return result


def save_result(result: dict, audio_path: str):
    """Save the result dict as a JSON file in OUTPUT_DIR."""
    stem      = Path(audio_path).stem
    out_file  = os.path.join(OUTPUT_DIR, f"{stem}_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {out_file}")
    return out_file


def main():
    # ── Validate INPUT_DIR ──────────────────────────────────────────────────
    if not os.path.isdir(INPUT_DIR):
        logger.error(
            f"INPUT_DIR not found: {INPUT_DIR}\n"
            f"Please create the folder and place your audio files in it."
        )
        return

    # ── Clean up leftover temp chunks from a previous crashed run ──────────
    chunks_tmp = os.path.join(INPUT_DIR, "_chunks_tmp")
    if os.path.isdir(chunks_tmp):
        logger.info(f"Cleaning up leftover temp chunks: {chunks_tmp}")
        shutil.rmtree(chunks_tmp, ignore_errors=True)

    # ── Create OUTPUT_DIR ───────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Input  directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # ── Collect audio files ────────────────────────────────────────────────
    audio_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not audio_files:
        logger.warning(
            f"No supported audio files found in {INPUT_DIR}.\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to process.\n")

    # ── Process each file ──────────────────────────────────────────────────
    total_files   = len(audio_files)
    success_count = 0
    skipped_count = 0
    failed_files  = []

    for file_num, audio_path in enumerate(audio_files, 1):
        # ── Skip if already processed ─────────────────────────────────
        stem     = Path(audio_path).stem
        out_file = os.path.join(OUTPUT_DIR, f"{stem}_result.json")
        if os.path.exists(out_file):
            logger.info(f"[{file_num}/{total_files}] SKIP (already processed): {Path(audio_path).name}")
            skipped_count += 1
            continue

        logger.info(f"[{file_num}/{total_files}] Starting: {Path(audio_path).name}")

        try:
            result   = process_audio_file(audio_path)
            out_path = save_result(result, audio_path)
            success_count += 1

            # Pretty-print segment count as a quick sanity check
            segs = result.get("transcriptions", {}).get("segments", [])
            logger.info(f"  Segments transcribed : {len(segs)}")
            logger.info(f"  Issue resolved       : {result.get('issue_resolved', 'N/A')}")
            logger.info(f"  Sentiment            : {result.get('sentiment', 'N/A')}")
            logger.info("")

        except Exception as e:
            import traceback
            logger.error(f"FAILED for {audio_path}: {e}")
            logger.error(traceback.format_exc())
            failed_files.append(audio_path)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"DONE  ✓ {success_count} succeeded   ⏭ {skipped_count} skipped   ✗ {len(failed_files)} failed")
    if failed_files:
        for f in failed_files:
            logger.error(f"  Failed: {f}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
