"""
============================================================
  STANDALONE DIARIZATION PROCESSOR
  (Test version — S3 and Kafka dependencies removed)
============================================================

This is a self-contained, local copy of gpt-new/diarization_gpt.py for
testing purposes only. It uses only the OpenAI API and local files.

Changes from production version:
  - REMOVED: S3 integration (boto3, bucket reads/writes)
  - REMOVED: Kafka integration
  - REMOVED: llama_index PydanticOutputParser
  - REMOVED: gpt.config / gpt.schemas / gpt.data_cleanup imports
  - ADDED  : Standalone noun_corpus fallback (read from local file if present)
  - ADDED  : Simple dict-based output (no Pydantic schema enforcement)
  - ADDED  : transcribe_with_openai drives the full pipeline locally

HOW TO USE:
  1. Create INPUT_DIR  (default: ./test_input)  and place audio files there.
  2. Create OUTPUT_DIR (default: ./test_output) — it is auto-created.
  3. Set OPENAI_API_KEY in a .env file at the project root.
  4. Run:  python diarization_test.py
"""

import os
import json
import time
import re
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Local ffmpeg path (project-based)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN = os.path.join(BASE_DIR, "ffmpeg", "bin")

# Add ffmpeg/bin to PATH so pydub's subprocess calls can find ffprobe
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Noun corpus — loaded from a local text file if present, else empty string
# ---------------------------------------------------------------------------
NOUN_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "gpt-new", "prompts", "noun_corpus.txt")
noun_corpus = ""
if os.path.exists(NOUN_CORPUS_PATH):
    with open(NOUN_CORPUS_PATH, "r", encoding="utf-8") as f:
        noun_corpus = f.read()


# ===========================================================================
#  STANDALONE DIARIZATION PROCESSOR
# ===========================================================================
class DiarizationProcessor:
    """
    Self-contained processor for diarization + field extraction.
    No S3, no Kafka, no Pydantic schemas.
    """

    def __init__(self, audio_path: str, logger=None):
        """
        Args:
            audio_path: Absolute path to the audio file to process.
            logger    : Optional Python logger.
        """
        self.logger = logger
        self._log("Initialising DiarizationProcessor (standalone / test mode).")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.client = OpenAI(api_key=api_key)
        self.model_name      = os.getenv("OPENAI_MODEL",      "gpt-4o")
        self.model_name_mini = os.getenv("OPENAI_MODEL_MINI", "gpt-4o-mini")

        # 2-minute chunking window (in milliseconds)
        # Reduced from 5 to 2 minutes to prevent JSON truncation
        self.AUDIO_CHUNKING_OFFSET = 120 * 1000

        self.audio_path = audio_path
        self.last_audio_transcript = ""   # set after transcription, used in extract_fields
        self.metadata = {}  # Set via setter or constructor if needed

        self._log(f"Primary model : {self.model_name}")
        self._log(f"Mini model    : {self.model_name_mini}")

        # Load production prompts if available
        self.prompts_dir = os.path.join(BASE_DIR, "gpt-new", "prompts")
        self.system_prompt = self._read_prompt("system.txt") or "You are an AI assistant."
        self.common_headers = self._read_prompt("common_headers.txt") or ""
        self.agent_metrics_prompt = self._read_prompt("agent_metrics_audio.txt")
        self.call_metrics_prompt = self._read_prompt("call_metrics_transcript.txt")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

    def _read_prompt(self, filename: str) -> str | None:
        path = os.path.join(self.prompts_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                self._log(f"Error reading prompt {filename}: {e}")
        return None

    def retry_with_backoff(self, func, max_retries=5, base_delay=15.0,
                           max_delay=300.0, *args, **kwargs):
        """Retry a callable with exponential back-off."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay)
                self._log(f"Attempt {attempt+1} failed ({e}). Retrying in {delay:.0f}s…")
                time.sleep(delay)

    def _extract_json(self, text: str) -> dict:
        """
        Robustly extract JSON from a string that might contain markdown markers
        or extra text before/after the actual JSON object.
        Includes repair logic for truncated JSON.
        """
        if not text:
            return {}

        # 1. Clean up potential markdown code blocks
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        cleaned = cleaned.strip()

        # 2. Try direct load
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3. Handle truncation - attempt to "repair" the JSON
        # This is common if the LLM hits output token limits
        repaired = cleaned
        
        # If it ends inside a string, close the quote
        if repaired.count('"') % 2 != 0:
            repaired += '"'
            
        # Add closing brackets in correct order
        stack = []
        for char in repaired:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
        
        # Close remaining brackets from inside out
        if stack:
            repaired += "".join(reversed(stack))
            
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 4. Find first { and last } (original method)
        try:
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_part = cleaned[start_idx : end_idx + 1]
                return json.loads(json_part)
        except json.JSONDecodeError:
            pass

        # 5. If all fails, raise original or custom error
        raise ValueError(f"Could not parse JSON from response. Content snippet: {cleaned[:200]}...{cleaned[-200:] if len(cleaned)>200 else ''}")

    # ------------------------------------------------------------------
    # Audio splitting
    # ------------------------------------------------------------------
    def split_audio(self, audio_path: str, audio: AudioSegment = None,
                    output_dir: str | None = None) -> dict:
        """
        Splits audio into chunks no longer than AUDIO_CHUNKING_OFFSET ms.
        Returns {chunk_index: local_file_path}.
        If `audio` is already loaded, reuses it to avoid a second ffmpeg decode.
        """
        ext = Path(audio_path).suffix.lower()
        fmt = "wav" if ext == ".wav" else "mp3"

        if audio is None:
            self._log(f"Loading audio: {audio_path}")
            audio = AudioSegment.from_file(audio_path, format=fmt)

        total      = len(audio)
        chunk_size = self.AUDIO_CHUNKING_OFFSET

        if output_dir is None:
            output_dir = str(Path(audio_path).parent)
        os.makedirs(output_dir, exist_ok=True)

        chunks = {}
        base   = Path(audio_path).stem
        if total <= chunk_size:
            chunks[0] = audio_path          # single chunk — use original
        else:
            idx = 0
            for start in range(0, total, chunk_size):
                segment   = audio[start:start + chunk_size]
                out_path  = os.path.join(output_dir, f"{base}_chunk_{idx}.{fmt}")
                segment.export(out_path, format=fmt)
                chunks[idx] = out_path
                idx += 1
                self._log(f"  Saved chunk {idx-1}: {out_path}")

        self._log(f"Audio split into {len(chunks)} chunk(s).")
        return chunks

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------
    def timestamp_to_seconds(self, ts: str) -> float:
        parts = ts.split(":")
        return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(ts)

    def seconds_to_timestamp(self, secs: float) -> str:
        m = int(secs // 60)
        s = secs - m * 60
        return f"{m:02d}:{s:06.3f}"

    def merge_json_with_offset(self, data: dict, offset_ms: int) -> list:
        """Merge chunk results, applying time offset per chunk index."""
        merged = []
        for idx in sorted(data.keys()):
            offset_sec = (idx * offset_ms) / 1000.0
            for seg in data[idx]:
                seg = dict(seg)
                try:
                    start_s = self.timestamp_to_seconds(seg["start"]) + offset_sec
                    end_s   = self.timestamp_to_seconds(seg["end"])   + offset_sec
                    seg["start"] = self.seconds_to_timestamp(start_s)
                    seg["end"]   = self.seconds_to_timestamp(end_s)
                except Exception:
                    pass
                merged.append(seg)
        return merged

    # ------------------------------------------------------------------
    # STEP 1 — gpt-4o-transcribe-diarize
    # ------------------------------------------------------------------
    def _gpt_transcribe_audio(self, audio_path: str) -> dict:
        """
        Calls gpt-4o-transcribe-diarize via the Audio Transcriptions API.
        Returns {'segments': [{start, end, text, speaker}, ...]}
        """
        self._log(f"[Step 1] Sending to gpt-4o-transcribe-diarize: {Path(audio_path).name}")
        self._log(f"  (this may take 30-90 seconds per chunk …)")
        t0 = time.time()
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto"
            )

        result = {"segments": []}
        segments_src = None

        if hasattr(response, "segments"):
            segments_src = response.segments
        elif isinstance(response, dict) and "segments" in response:
            segments_src = response["segments"]

        if segments_src:
            for seg in segments_src:
                # Handle both object attributes and dict keys
                get = seg.get if isinstance(seg, dict) else lambda k, d=None: getattr(seg, k, d)
                result["segments"].append({
                    "start":   get("start"),
                    "end":     get("end"),
                    "text":    get("text"),
                    "speaker": get("speaker"),
                })

        self._log(f"  → {len(result['segments'])} raw segments returned ({time.time()-t0:.1f}s).")
        return result

    # ------------------------------------------------------------------
    # STEP 2 — gpt-4o-mini: Translation + Role Mapping + Timestamp Format
    # ------------------------------------------------------------------
    def transcribe_chunk(self, idx: int, chunk_path: str,
                         past_reference: str | None = None):
        """
        Runs Step 1 then Step 2 for a single audio chunk.
        Returns (idx, segments_list).
        """
        # ── STEP 1 ──────────────────────────────────────────────────────
        gpt_audio_result = self.retry_with_backoff(
            self._gpt_transcribe_audio, audio_path=chunk_path
        )
        self._log(f"Chunk {idx}: Step 1 complete.")
        raw_segments = gpt_audio_result.get("segments", [])

        # ── STEP 2 ──────────────────────────────────────────────────────
        self._log(f"Chunk {idx}: Step 2 — translation + role mapping.")

        system_prompt = f"""
        **IMPORTANT: You are in a call centre for a brand called Dish Tv, where your job is to transcribe the call recordings.**
        **IMPORTANT: You are operating in India only, so you need to use the ₹ symbol for rupees. Wherever you encounter any currency reference, you must use the ₹ symbol (Indian Rupee symbol) instead of any other currency symbols.**
        The following is a list of nouns and terms that are commonly used in the call recordings:
        {noun_corpus}
        Please make sure to use the correct noun and term in the transcription.
        """

        prompt = f"""
        **IMPORTANT : I want the translation in english language Only.**\n\n\n
        **Make sure wherever the brand name Dish TV is mentioned, you should use the correct brand name. please do not mis-spell or convert it to any other name.**

        Your role is to identify the speaker in the call like agent, customer, technician, etc.
        Identify and assign unique speaker labels (e.g., agent, customer, senior agent, technician, chief technician) to each distinct voice in the call.
        
        **IMPORTANT: You must assign a unique speaker label to each distinct voice in the call.**
        Different voices can be distinguished by variations in pitch, frequency, intensity, timbre, speaking rate, accent, and vocal characteristics.
        Pay careful attention to voice changes that may indicate a new speaker joining the conversation.
        Maintain consistent speaker identification on the basis of the voice characteristics and context throughout the entire conversation - once a speaker is labeled, use the same label for all their segments.
        
        Agent means a person who is representing the brand and play a central role in providing service to the customer. And interact with chief technician or technician to arrange the on ground service.
        Customer means a person who wants to resolve their issue.
        Technician means a person who is representing the brand and trying to resolve the issue by providing the on ground service.
        **Chief Technician means a senior technician who manages other technicians and coordinates between customers and agents.**
        Recorded audio means any recorded audio(music, song, caller tune, ring tone, background music, voice mail, etc.) in the call.
        When additional participants join the call, they should be identified as senior agent, technician, chief technician, or other call center personnel based on their role and context in the conversation.

        Senior agent means a person who is representing the brand and play a central role in providing further assistance to the customer as current agent is not able to resolve the issue. Generally senior agent to transfer the call to them.
        
        Translate this audio to English language only.
        Avoid use of native language words - only English words are needed. Convert all non-English words to their English equivalents.
        Ensure to caption all the words in the audio (Do not ignore any word).
        Ensure proper punctuation of the sentences.
        Also break the sentence when the speaker changes.
        If nothing is spoken at certain time intervals then ignore those time intervals.

        Return the output strictly as a JSON object using the below format:
        {{
            "segments": [
                {{
                    "start": "MM:SS.MS",
                    "end":   "MM:SS.MS",
                    "text":  "Translation in english script",
                    "speaker": "agent / customer / technician / etc."
                }}
            ]
        }}
        **IMPORTANT: Time Start and End MUST be in MM:SS.MS format only (e.g. "00:25.452"). Do NOT use HH:MM:SS.MS.**

        RAW TRANSCRIPTION:
        {json.dumps(raw_segments, indent=2)}
        """

        if past_reference:
            prompt += f"\n\n{past_reference}"

        response = self.client.chat.completions.create(
            model=self.model_name_mini,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        try:
            result_data = self._extract_json(raw_content)
        except Exception as e:
            self._log_error(f"Chunk {idx}: JSON parsing failed.")
            self._log_error(f"Raw response head: {raw_content[:500]}")
            self._log_error(f"Raw response tail: {raw_content[-500:]}")
            raise

        json_data = result_data.get("segments", [])
        self.last_audio_transcript = json.dumps(json_data, indent=2)
        self._log(f"Chunk {idx}: Step 2 complete — {len(json_data)} segments.")
        return idx, json_data

    # ------------------------------------------------------------------
    # STEP 3 — gpt-4o-mini: Field Extraction
    # ------------------------------------------------------------------
    def extract_fields(self, prompt: str, system_instruction: str = None) -> dict:
        """
        Runs Step 3: extract business metrics.
        Returns a dict of extracted fields.
        """
        self._log("[Step 3] Extracting fields with gpt-4o-mini …")
        t0 = time.time()
        
        # Ensure transcript is included
        full_prompt = f"{prompt}\n\nTranscript:\n{self.last_audio_transcript}"
        
        messages = []
        if system_instruction:
            # Production uses first part of system prompt
            sys_instr = system_instruction.split("\n\n\n")[0] if "\n\n\n" in system_instruction else system_instruction
            messages.append({"role": "system", "content": sys_instr})
        
        messages.append({"role": "user", "content": full_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name_mini,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        self._log(f"Step 3 extraction complete ({time.time()-t0:.1f}s).")
        try:
            return self._extract_json(content)
        except Exception as e:
            self._log_error(f"Step 3: JSON parsing failed: {e}")
            return {"raw_response": content}

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def transcribe_with_openai(self) -> list:
        """
        Runs the full Step 1 + Step 2 pipeline.
        Chunks long audio automatically.
        Returns a flat list of translated + labelled segments.
        """
        # Load audio ONCE — reused for duration check AND splitting
        self._log(f"Loading audio file: {Path(self.audio_path).name}")
        ext = Path(self.audio_path).suffix.lower()
        fmt = "wav" if ext == ".wav" else "mp3"
        audio    = AudioSegment.from_file(self.audio_path, format=fmt)
        duration = len(audio)   # ms
        self._log(f"  Duration: {duration/1000:.1f}s")

        if duration <= self.AUDIO_CHUNKING_OFFSET:
            self._log("Short audio — single-chunk mode.")
            _, segments = self.transcribe_chunk(0, self.audio_path)
            return segments
        else:
            self._log("Long audio — multi-chunk mode.")
            tmp_dir = os.path.join(os.path.dirname(self.audio_path), "_chunks_tmp")
            # Pass the already-loaded audio to avoid a second ffmpeg decode
            chunks  = self.split_audio(self.audio_path, audio=audio, output_dir=tmp_dir)
            results = {}
            previous_chunk_text = ""

            for idx, chunk_path in sorted(chunks.items()):
                self._log(f"  Processing chunk {idx+1}/{len(chunks)} …")
                past_ref = f"Previous chunk: {previous_chunk_text}" if previous_chunk_text else None
                idx, json_data = self.transcribe_chunk(idx, chunk_path, past_ref)
                results[idx]   = json_data
                if json_data:
                    previous_chunk_text = " ".join(s.get("text", "") for s in json_data)
                # Remove temp chunk (skip if it was the original file)
                if chunk_path != self.audio_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)

            # Remove tmp dir if empty
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

            return self.merge_json_with_offset(results, self.AUDIO_CHUNKING_OFFSET)

    def process(self, metadata: dict = None) -> dict:
        """
        Full pipeline:
          1. Transcription + diarization (Steps 1 & 2)
          2. Field extraction           (Step 3)

        Returns:
            {
              "transcriptions": { "segments": [...] },
              ... field-extraction keys ...
            }
        """
        if metadata:
            self.metadata = metadata

        # Steps 1 & 2
        segments = self.transcribe_with_openai()

        output = {
            "audio_path":      self.audio_path,
            "transcriptions": {"segments": segments},
        }

        # Step 3 — Comprehensive Extraction
        if self.agent_metrics_prompt and self.call_metrics_prompt:
            self._log("Using production prompts for field extraction.")
            
            # 1. Agent Metrics
            common = self.common_headers.format(metadata=self.metadata) if self.common_headers else ""
            agent_prompt = common + self.agent_metrics_prompt
            agent_results = self.extract_fields(agent_prompt, self.system_prompt)
            output.update(agent_results)

            # 2. Call Metrics
            # Note: Call metrics template might have {metadata} and {transcriptions} placeholders
            try:
                call_prompt = self.call_metrics_prompt.format(
                    metadata=self.metadata, 
                    transcriptions=json.dumps(segments, indent=2)
                )
            except Exception:
                call_prompt = self.call_metrics_prompt # fallback
            
            call_results = self.extract_fields(call_prompt, self.system_prompt)
            output.update(call_results)
        else:
            self._log("Production prompts not found. Using basic extraction.")
            field_prompt = """
            You are an AI quality analyst for a Dish TV call centre.
            Based on the transcript provided, extract the following fields as a JSON object:
            - call_summary       (string: 1-2 sentence summary)
            - issue_resolved     (yes / no / partial)
            - speakers_identified (list of speaker roles present)
            - language_used      (predominant language before translation)
            - sentiment          (positive / neutral / negative)
            - total_segments     (integer)
            Return ONLY a JSON object with these keys.
            """
            extracted = self.extract_fields(
                prompt=field_prompt,
                system_instruction="You are a JSON-only responder. Never include markdown or extra text."
            )
            output.update(extracted)

        return output
