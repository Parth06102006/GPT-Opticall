"""
Module for audio diarization and transcription using OpenAI GPT-4o Audio + GPT-4o.

Approach: Two-step pipeline using GPT models only (no Whisper):
  Step 1: GPT-4o-audio-preview accepts audio input natively → raw transcription
  Step 2: GPT-4o / GPT-4o-mini processes the raw transcript → diarization,
          speaker identification, English translation, and field extraction

Supports multiple Indian languages via GPT's multilingual capabilities.
"""
import os
import json
import time
import base64
from pathlib import Path
import re
import random
import tempfile
from openai import OpenAI

from gpt.data_cleanup import clean_data
from llama_index.core.output_parsers import PydanticOutputParser

from pydub import AudioSegment

from concurrent.futures import ThreadPoolExecutor, as_completed
from gpt.config import (
    SYSTEM_PROMPT_PATH,
    USER_PROMPT_COMMON_HEADERS_PATH,
    USER_PROMPT_AGENT_CUSTOMER_PATH,
    USER_PROMPT_AGENT_METRICS_AUDIO_PATH,
    USER_PROMPT_CALL_METRICS_TRANSCRIPT_PATH,
    dish_script,
    d2h_script,
    noun_corpus,
    s3_client,
    bucket_name)
from gpt.schemas import (AgentMetricsAudio,
    CallMetricsTranscriptResponse,
    TranscriptionsResponse)


class DiarizationProcessor:
    def __init__(self, message, metadata_path, audio_path, logger=None):
        """Initialize the DiarizationProcessor with OpenAI GPT-4o Audio + GPT-4o configuration."""
        self.logger = logger
        if self.logger:
            self.logger.info("Initializing DiarizationProcessor with GPT-4o-audio-preview + GPT-4o (two-step)")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # GPT-4o for text analysis (diarization, field extraction)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        # Mini variant for cost-effective operations
        self.model_name_mini = os.getenv("OPENAI_MODEL_MINI", "gpt-4o-mini")
        
        if self.logger:
            self.logger.info(f"Using primary model: {self.model_name}")
            self.logger.info(f"Using mini model: {self.model_name_mini}")
        
        self.AUDIO_CHUNKING_OFFSET = 3000
        self.call_id = message["call_id"]
        self.divisions = list(message["call_divisions"])
        
        self.transcriptions = []
        self.metadata_path = [metadata_path] if isinstance(metadata_path, str) else metadata_path
        self.audio_path = [audio_path] if isinstance(audio_path, str) else audio_path
        self.output_parser = {
            "agent-metrics-audio": PydanticOutputParser(AgentMetricsAudio),
            "call-metrics-transcript": PydanticOutputParser(CallMetricsTranscriptResponse),
            "transcriptions": PydanticOutputParser(TranscriptionsResponse),
        }
        self.model_classes = {
            "agent-metrics-audio": AgentMetricsAudio,
            "call-metrics-transcript": CallMetricsTranscriptResponse,
        }
        self.user_prompts = {
            "agent-metrics-audio": self.read_user_prompt_from_file(USER_PROMPT_AGENT_METRICS_AUDIO_PATH),
            "call-metrics-transcript": self.read_user_prompt_from_file(USER_PROMPT_CALL_METRICS_TRANSCRIPT_PATH),
            "common-headers": self.read_user_prompt_from_file(USER_PROMPT_COMMON_HEADERS_PATH)
        }
        self.system_prompt = ""
        # Store the last GPT audio transcript for reuse in extract_fields
        self.last_audio_transcript = ""
        try:
            self.system_prompt = self.read_prompt_from_file(SYSTEM_PROMPT_PATH).split("\n\n\n")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading system prompt: {str(e)}")
            raise
    
    def read_user_prompt_from_file(self, file_path: str) -> str:
        """
        Read a user prompt template from a text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
        return prompt_template

    def read_prompt_from_file(self, file_path: str) -> str:
        """
        Read a prompt template from a text file.
        
        Args:
            file_path (str): Path to the text file containing the prompt template
            
        Returns:
            str: The prompt template as a string
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            Exception: For other file reading errors
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt_template = file.read().strip()
            return prompt_template
        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"Prompt template file not found: {file_path}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading prompt template file: {str(e)}")
            raise
    
    def get_script_prompt(self, metadata):
        """Get the script prompt for the given metadata."""
        if metadata["brand"].lower() == "dish":
            try:
                script_dict = dish_script[dish_script['language'] == metadata['call_language'].upper()].iloc[0].to_dict()
            except IndexError:
                script_dict = dish_script[dish_script['language'] == 'ENGLISH'].iloc[0].to_dict()
            scripts=f"""These are the reference scripts for evaluating the call:
                {script_dict}

                Important Notes:
                - These are guidelines for evaluation, not verbatim scripts that must be followed
                - The agent should convey the key information naturally in their own words
                - Focus on whether the essential points were communicated effectively
                - Ignore exact matching of advisor/agent names
                - Evaluate based on adherence to core message and professional delivery
            """
            return scripts
        elif metadata["brand"].lower() == "d2h":
            try:
                script_dict = d2h_script[d2h_script['language'] == metadata['call_language'].upper()].iloc[0].to_dict()
            except IndexError:
                script_dict = d2h_script[d2h_script['language'] == 'ENGLISH'].iloc[0].to_dict()
            
            scripts=f"""These are the reference scripts for evaluating the call:
                {script_dict}

                Important Notes:
                - These are guidelines for evaluation, not verbatim scripts that must be followed
                - The agent should convey the key information naturally in their own words
                - Focus on whether the essential points were communicated effectively
                - Ignore exact matching of advisor/agent names
                - Evaluate based on adherence to core message and professional delivery
            """
            return scripts
        else:
            return ""
        
    def split_audio(self, audio_path, output_dir=None):
        """
        Splits the audio file into chunks and saves them locally.
        Handles both WAV and MP3 formats.
        
        Args:
            audio_path: Path to the local audio file
            output_dir: Directory to save the chunks (default: same directory as input file)
        
        Returns:
            A dictionary mapping chunk numbers to local file paths
        """
        if self.logger:
            self.logger.info(f"Splitting audio file: {audio_path}")
        
        # Validate input file exists
        if not os.path.exists(audio_path):
            if self.logger:
                self.logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
        AUDIO_OFFSET_CHUNK = int(os.getenv("AUDIO_OFFSET_CHUNK", 300))
        if self.logger:
            self.logger.info(f"Using chunk offset: {AUDIO_OFFSET_CHUNK} seconds")
        
        # Convert seconds to milliseconds for pydub
        chunk_duration_ms = AUDIO_OFFSET_CHUNK * 1000
        
        # Determine file format
        file_name = os.path.basename(audio_path)
        file_extension = file_name.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            if self.logger:
                self.logger.error(f"Unsupported audio format: {file_extension}")
            raise ValueError("Unsupported audio format. Only WAV and MP3 are supported.")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Loading audio file: {audio_path}")
        # Load the audio file
        audio = AudioSegment.from_file(audio_path, format=file_extension)
        
        chunks = []
        total_chunks = len(audio) // chunk_duration_ms + (1 if len(audio) % chunk_duration_ms else 0)
        if self.logger:
            self.logger.info(f"Creating {total_chunks} chunks")
        
        for i, start_time in enumerate(range(0, len(audio), chunk_duration_ms)):
            if self.logger:
                self.logger.debug(f"Processing chunk {i+1}/{total_chunks}")
            # Convert 10 seconds to milliseconds for overlap
            overlap_ms = 0 * 1000
            chunk = audio[max(0, start_time - overlap_ms):start_time + chunk_duration_ms]
            
            # Save the chunk in WAV format
            base_name = os.path.splitext(file_name)[0]
            chunk_file_name = f"{base_name}_chunk{i}.wav"
            chunk_path = os.path.join(output_dir, chunk_file_name)
            chunk.export(chunk_path, format="wav")
            
            chunks.append((i, chunk_path))
        
        return dict(chunks)
    

    def retry_with_backoff(self, func, max_retries=5, base_delay=15.0, max_delay=300.0, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = min(max_delay, base_delay * (2 ** attempt))
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                time.sleep(total_delay)

    def timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert a timestamp string like 'MM:SS.mmm' to seconds."""
        minutes, rest = timestamp.split(":")
        seconds = float(rest)
        return int(minutes) * 60 + seconds

    def seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp string like 'MM:SS.mmm'."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def merge_json_with_offset(self, data, time_offset):
        """Merge multiple JSON arrays with time offset."""
        merged_array = []
        sorted_data = sorted(data.items(), key=lambda x: x[0])

        for i, json_array in sorted_data:
            offset_seconds = i * time_offset
            for entry in json_array:
                new_entry = entry.copy()
                new_entry['start'] = (self.timestamp_to_seconds(entry['start']) + offset_seconds)
                new_entry['end'] = (self.timestamp_to_seconds(entry['end']) + offset_seconds)
                merged_array.append(new_entry)

        return merged_array

    def _gpt_transcribe_audio(self, audio_path):
        """
        Step 1: Use gpt-4o-transcribe-diarize to transcribe and diarize audio.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            dict: Response with raw transcription segments including speaker labels
        """
        if self.logger:
            self.logger.info(f"[Step 1/3] GPT transcription and diarization: {audio_path}")
        
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="verbose_json"
            )
        
        # Verbose JSON gives us segments with start, end, text, and speaker
        result = {"segments": []}
        if hasattr(response, 'segments'):
            for seg in response.segments:
                result["segments"].append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "speaker": seg.get("speaker")
                })
        elif isinstance(response, dict) and "segments" in response:
            for seg in response["segments"]:
                result["segments"].append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "speaker": seg.get("speaker")
                })

        if self.logger:
            seg_count = len(result.get("segments", []))
            self.logger.info(f"gpt-4o-transcribe-diarize returned {seg_count} segments")
        
        return result

    def call_model(self, content, extract_model=None, generation_config=None, model_name=None, system_instruction=None):
        """
        Call the OpenAI GPT-4o model to generate content.
        Uses Chat Completions API (text only — no audio input).
        
        Args:
            content: The text content to process
            extract_model: The extraction model type
            generation_config: dict with generation parameters (temperature, etc.)
            model_name: The model name to use (defaults to self.model_name)
            system_instruction: System instruction for the model
            
        Returns:
            The model's response
        """
        if model_name is None:
            model_name = self.model_name
        
        temperature = 0.2
        if generation_config and isinstance(generation_config, dict):
            temperature = generation_config.get("temperature", 0.2)
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        # GPT-4o: text-only input
        if isinstance(content, list):
            # Flatten to text-only content
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item["text"])
            messages.append({"role": "user", "content": "\n\n".join(text_parts)})
        else:
            messages.append({"role": "user", "content": content})
            
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        return response

    def transcribe_chunk(self, idx, chunk_path, past_reference=None):
        """
        Transcribe a single audio chunk using the new 3-step approach:
        
        Step 1: gpt-4o-transcribe-diarize -> raw transcription + speaker_X labels
        Step 2: gpt-4o-mini -> translation + semantic speaker mapping + timestamp formatting
        """
        
        # ============================================================
        # STEP 1: gpt-4o-transcribe-diarize
        # ============================================================
        gpt_audio_result = self._gpt_transcribe_audio(chunk_path)
        
        if self.logger:
            self.logger.info(f"Chunk {idx}: Step 1 (transcribe-diarize) complete")
        
        raw_segments = gpt_audio_result.get("segments", [])
        
        # ============================================================
        # STEP 2: GPT-4o-mini — Translation + Semantic Role Mapping + Formatting
        # ============================================================
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
        When additional participants join the call, they should be identified as senior agent,technician, chief technician, or other call center personnel based on their role and context in the conversation.

        Senior agent means a person who is representing the brand and play a central role in providing further assistance to the customer as current agent is not able to resolve the issue. Generally senior agent to transfer the call to them.
        
        Translate this audio to English language only.
        Avoid use of native language words - only English words are needed. Convert all non-English words to their English equivalents.
        Ensure to caption all the words in the audio (Do not ignore any word).
        Ensure proper punctuation of the sentences.
        Also break the sentence when the speaker changes.
        If nothing is spoken at certain time intervals then ignore those time intervals.

        Return the output strictly as a JSON array of objects using the below format:
        ```json
        {{"segments":
            [
                {{
                "start": "MM:SS.MS" in string, Example: "00:25.452"
                "end": "MM:SS.MS" in string, Example: "01:01.197"
                "text": "Translation in english script",
                "speaker": "agent", "customer", "technician", "etc."
                }}
            ]
        }}
        ```
        **IMPORTANT: Make Sure Time Start and End are in the format of MM:SS.MS only. Do not try to be over smart and convert it to HH:MM:SS.MS. or any other format.**
        Only return the pure JSON array inside triple backticks and nothing else.

        RAW TRANSCRIPTION:
        {json.dumps(raw_segments, indent=2)}
        """
        
        if past_reference:
            prompt += f"\n\n{past_reference}"

        response = self.client.chat.completions.create(
            model=self.model_name_mini,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result_data = json.loads(content)
        json_data = result_data.get("segments", [])
        
        # Store translated diarized segments for field extraction
        self.last_audio_transcript = json.dumps(json_data, indent=2)
        
        if self.logger:
            self.logger.info(f"Chunk {idx}: Step 2 (mini processing) complete with {len(json_data)} segments")
        
        return idx, json_data

    def transcribe_chunks(self, audio_uri):
        """Transcribe multiple audio chunks sequentially with previous chunk reference."""
        chunks_dict = self.split_audio(audio_uri)
        results = {}
        
        if self.logger:
            self.logger.info(f"Processing {len(chunks_dict)} chunks sequentially with previous chunk reference")

        # Sort chunks by index to maintain order
        sorted_chunks = sorted(chunks_dict.items(), key=lambda x: x[0])
        
        # Keep track of previous chunk's transcription for reference
        previous_chunk_text = ""
        
        for idx, chunk_uri in sorted_chunks:
            try:
                # Pass the previous chunk's transcription as reference
                past_reference = previous_chunk_text if previous_chunk_text else None
                idx, json_data = self.transcribe_chunk(idx, chunk_uri, past_reference)
                results[idx] = json_data
                
                # Update previous chunk text for next iteration
                if json_data:
                    # Convert current chunk's transcription to text for future reference
                    chunk_text = " ".join([segment.get("text", "") for segment in json_data])
                    if chunk_text:
                        previous_chunk_text = f"Previous chunk transcription: {chunk_text}"
                
                if self.logger:
                    self.logger.info(f"Successfully processed chunk {idx}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing chunk {idx}: {str(e)}")
                raise

        final_json = self.merge_json_with_offset(results, self.AUDIO_CHUNKING_OFFSET)

        # Remove all chunk files
        for chunk_path in chunks_dict.values():
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error removing chunk file {chunk_path}: {str(e)}")
        return final_json

    def transcribe_with_openai(self, audio_path, duration):
        """Transcribe audio using GPT-4o-audio-preview + GPT-4o pipeline."""
        if duration <= self.AUDIO_CHUNKING_OFFSET:
            idx, transcription = self.transcribe_chunk(0, audio_path)
            return transcription
        else:
            transcription = self.transcribe_chunks(audio_path)
            return transcription

    # Keep backward compatibility alias
    transcribe_with_gemini = transcribe_with_openai

    def get_segments(self, audio_path):
        """Get all segments from audio file."""
        # Function to check if a file exists in an S3 bucket and return its S3 path if present
        def get_s3_file_if_exists(s3_client, bucket_name, file_key):
            """
            Checks if a file exists in an S3 bucket and returns its S3 path if present.

            Args:
                s3_client: boto3 S3 client object.
                bucket_name (str): Name of the S3 bucket.
                file_key (str): The key (path/filename) to check.

            Returns:
                dict: The segments data from S3.
                
            Raises:
                FileNotFoundError: If the file does not exist in S3.
            """
            try:
                s3_client.head_object(Bucket=bucket_name, Key=file_key)
                response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                body = response["Body"].read().decode('utf-8')
                return json.loads(body)["segments"]
            except Exception as e:
                raise FileNotFoundError(f"Required transcript file not found in S3: {file_key}") from e
        
        ucid = os.path.splitext(os.path.basename(audio_path))[0]
        
        procesed_transcripts = get_s3_file_if_exists(s3_client, bucket_name, f"transcripts-batch/{ucid}.json")
        print(f"Processed transcripts found for UCID: {ucid}")
        print(procesed_transcripts)
        return procesed_transcripts

    def extract_fields(self, instruction, system_prompt, extract_model, mode="transcript", audio_path=None):
        """
        Extract fields using GPT-4o-mini (text-only).
        Always prepends the stored transcript to the prompt.
        """
        if extract_model == "agent-metrics-audio":
            # Existing S3 logic remains unchanged as per "DO NOT MODIFY" rules
            def get_s3_file_if_exists(s3_client, bucket_name, file_key):
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=file_key)
                    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    body = response["Body"].read().decode('utf-8')
                    return json.loads(body)
                except Exception as e:
                    raise FileNotFoundError(f"Required agent metrics audio file not found in S3: {file_key}") from e
            
            ucid = os.path.splitext(os.path.basename(audio_path))[0]
            procesed_agent_metrics_audio = get_s3_file_if_exists(s3_client, bucket_name, f"agent-metrics-audio-batch/{ucid}.json")
            return procesed_agent_metrics_audio
        
        # Step 3: Field extraction using gpt-4o-mini (text only)
        # Always append the transcript
        full_instruction = f"{instruction}\n\nTranscript:\n{self.last_audio_transcript}"
        full_prompt = f"{system_prompt}\n\n{full_instruction}"
        
        messages = [{"role": "user", "content": full_prompt}]
            
        response = self.client.chat.completions.create(
            model=self.model_name_mini,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.lower()
        extract_data = self.output_parser[extract_model].parse(content)
        
        return extract_data

    def convert_to_boolean(self, value):
        """Convert string boolean values to proper boolean values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.lower()
            if value in ['yes', 'true', '1', 'y']:
                return True
            elif value in ['no', 'false', '0', 'n']:
                return False
        return None

    def process_diarization(self):
        """Process audio file for diarization and transcription."""
        output_data=[]
        for i, data in enumerate(self.divisions):
            information_extracted = {}
            
            all_segments = self.get_segments(self.audio_path[i])

            transcriptions = []

            for segment in all_segments:
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Convert HH:MM:SS.MS to seconds
                if isinstance(start_time, str):
                    if ":" in start_time:
                        start_parts = start_time.split(":")
                        if len(start_parts) == 3:  # HH:MM:SS.MS format
                            start_seconds = float(start_parts[0]) * 3600 + float(start_parts[1]) * 60 + float(start_parts[2])
                        else:  # MM:SS.MS format
                            start_seconds = float(start_parts[0]) * 60 + float(start_parts[1])
                    else:
                        start_seconds = float(start_time)
                else:
                    start_seconds = (start_time)
                    
                if isinstance(end_time, str):
                    if ":" in end_time:
                        end_parts = end_time.split(":")
                        if len(end_parts) == 3:  # HH:MM:SS.MS format
                            end_seconds = float(end_parts[0]) * 3600 + float(end_parts[1]) * 60 + float(end_parts[2])
                        else:  # MM:SS.MS format
                            end_seconds = float(end_parts[0]) * 60 + float(end_parts[1])
                    else:
                        end_seconds = float(end_time)
                else:
                    end_seconds = (end_time)

                segment["start"] = start_seconds
                segment["end"] = end_seconds

                transcriptions.append(segment)
            
            information_extracted["transcriptions"]=transcriptions
            # Also store as last_audio_transcript for field extraction context
            self.last_audio_transcript = json.dumps(transcriptions, indent=2)
            
            with open(self.metadata_path[i], "r") as f:
                metadata = json.load(f)
                f.close()
            
            self.transcriptions.append(transcriptions)
            tasks=[]
            common_headers = self.user_prompts["common-headers"].format(metadata = metadata)
            
            if i==len(self.divisions)-1:
                tasks.extend([
                    (common_headers + self.user_prompts["agent-metrics-audio"], (self.system_prompt[0]), "agent-metrics-audio", "audio", self.audio_path[i]),
                    (self.user_prompts["call-metrics-transcript"].format(metadata = metadata, transcriptions = self.transcriptions), "\n\n\n".join(self.system_prompt[0:3]), "call-metrics-transcript", "transcript", None)
                ])
            else:
                tasks.extend([
                    (common_headers + self.user_prompts["agent-metrics-audio"], (self.system_prompt[0]), "agent-metrics-audio", "audio", self.audio_path[i])
                ])

            if self.logger:
                self.logger.info("Starting field extraction tasks")
            
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_lang = {
                    executor.submit(self.extract_fields, i[0], i[1], i[2], i[3], i[4]): i 
                    for i in tasks
                }
                
                for future in as_completed(future_to_lang):
                    try:
                        extracted_data = future.result()
                        information_extracted.update(extracted_data)
                        if self.logger:
                            self.logger.info("Successfully extracted fields")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error during field extraction: {str(e)}")
                        raise
            
            # Convert all string values in output_data to lowercase and handle boolean conversions
            for key, value in information_extracted.items():
                if isinstance(value, str):
                    information_extracted[key] = value.lower()
                elif isinstance(value, list):
                    information_extracted[key] = [item.lower() if isinstance(item, str) else item for item in value]
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            value[k] = v.lower()
                        elif isinstance(v, list):
                            value[k] = [item.lower() if isinstance(item, str) else item for item in v]
            
            # Convert boolean string values to proper boolean values
            boolean_fields = [
                'agent_handling_rebuttals', 'agent_telephonic_etiquette', 'agent_empathy',
                'agent_active_listening', 'agent_correct_probing', 'hold_script_used',
                'issue_resolved', 'agent_commit_for_resolution', 'follow_up_required',
                'first_call_resolution', 'repeated_calls', 'field_request_asked',
                'agent_ownship', 'agent_anger_handling', 'customer_confirmation_for_resolution',
                'call_dropped', 'call_escalated', 'offer_pitched', 'is_offer_purchased',
                'promotion_of_ott', 'recharge_accepted', 'technician_competence',
                'charges_customer_refused', 'waivers_requested', 'waivers_offered',
                'promotion_of_products', 'profanity', 'competitor_mentioned',
                'multiple_call_mentioned', 'language_dialect_issue',
                'agent_sentence_construction_quality', 'agent_pronunciation_and_fluency',
                'customer_know_the_error_code', 'promise_of_payment', 'free_of_cost'
            ]
            
            for field in boolean_fields:
                if field in information_extracted:
                    information_extracted[field] = self.convert_to_boolean(information_extracted[field])
            
            information_extracted.update(metadata)
            # Format transcriptions data for the Transcriptions model
            information_extracted["transcriptions"] = {"segments": transcriptions}
            if self.logger:
                self.logger.info("Diarization process completed successfully")
            
            output_data.append(information_extracted)

        if self.logger:
            self.logger.info("Cleaning data")
        print(output_data)
        output_data=clean_data(output_data)
        
        return output_data

if __name__ == "__main__":
    import sys
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Default to an audio file that exists in the audio directory
    default_audio = os.path.join(project_root, "audio", "31162886855553918.mp3")
    test_audio_path = sys.argv[1] if len(sys.argv) > 1 else default_audio
    
    # Verify the audio file exists
    if not os.path.exists(test_audio_path):
        print(f"ERROR: Audio file not found: {test_audio_path}")
        print(f"Please provide a valid audio file path as an argument or ensure the default file exists.")
        sys.exit(1)
    
    # Use a temp directory in the project root or current directory
    temp_dir = os.path.join(project_root, "temp") if os.path.exists(project_root) else os.getcwd()
    os.makedirs(temp_dir, exist_ok=True)
    test_output_json = os.path.join(temp_dir, f"{Path(test_audio_path).stem}_diarization.json")
    audio_path = [test_audio_path]
    metadata_path = [test_output_json]
    message={
        "call_id": "V71BD82QQ56NBFAPR5SRDMCDR8019F3B",
        "call_divisions": [
            {
                "ucid": "31162886855553918",
                "s3_audio_path": "cleaned-audio-recordings-ucid/31162886855553918.mp3",
                "level": "L1"
            }
        ]
    }
    
    try:
        start_time = time.time()
        processor = DiarizationProcessor(message=message, audio_path=test_audio_path, metadata_path=test_output_json)
        result = processor.process_diarization()
        
        elapsed_time = time.time() - start_time
        print('-'*50)
        print(f" DIARIZATION TEST COMPLETED in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f" ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
