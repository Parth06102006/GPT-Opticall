"""
GPT Pipeline Processor - Processes audio files using OpenAI GPT for diarization.

Same S3 download + database flow as the original processor.py, but:
  - Imports DiarizationProcessor from gpt.diarization_gpt (OpenAI-based)
  - No Gemini/GCP usage
"""
import sys
import os

# Ensure project root is in sys.path for src.* imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gpt.diarization_gpt import DiarizationProcessor
from gpt.config import s3_client, bucket_name, login_api, opticall_dataframes_api, audio_folder, metadata_folder, login_data, DATABASE_URL
from gpt.logger import create_logger
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from gpt.schemas import CallRecord, Base
from sqlalchemy import text
from contextlib import contextmanager
import json

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
logger = create_logger("logs/gpt_processor.log")

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    engine = create_engine(DATABASE_URL)
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
    finally:
        session.close()
        engine.dispose()

def process_call(message, logger=logger):
    """
    Process audio file using OpenAI GPT for diarization and transcription.
    
    Pipeline:
      1. Download audio + metadata from S3
      2. Run DiarizationProcessor (GPT-4o-audio-preview + GPT-4o)
      3. Save results to database
    
    Args:
        message (dict): Message containing call information and S3 paths
        logger: Logger instance for logging
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    audio_path = None
    metadata_path = None
    session = None
    
    try:
        logger.info("="*50)
        logger.info("Starting new call processing (GPT pipeline)")
        logger.info(f"Input message: {json.dumps(message, indent=2)}")
        call_id = message.get('call_id')
        final_metadata_path = []
        final_audio_path = []
        try:
            with get_db_session() as session:
                # Check if call_id exists in scheduler
                query = text("SELECT * FROM scheduler WHERE call_id = :call_id")
                result = session.execute(query, {"call_id": call_id}).fetchone()
                print(result)
                if result and result.call_status == 'completed':
                    logger.info(f"Call ID {call_id} exists in scheduler and is completed")
                    return True
                elif not result:
                    logger.info(f"Call ID {call_id} not found in scheduler or is not completed")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Database error while checking call_id: {str(e)}")
            return False

        for i in message.get('call_divisions'):
            s3_audio_path = i.get('s3_audio_path')
            ucid = i.get('ucid')
            logger.info(f"Processing ucid: {ucid}")
            logger.info(f"S3 audio path: {s3_audio_path}")
            
            if not s3_audio_path:
                logger.error("Validation failed: s3_audio_path not found in message")
                return False
                
            if not ucid:
                logger.error("Validation failed: ucid not found in message")
                return False
                
            # Extract filename from s3 path
            file_name = os.path.basename(s3_audio_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            logger.info(f"File name: {file_name}")
            logger.info(f"File name without extension: {file_name_without_ext}")
                
            # Download file from S3
            audio_path = os.path.join(audio_folder, file_name)
            s3_metadata_path = f"metadata-ucid/{file_name_without_ext}.json"
            metadata_path = os.path.join(metadata_folder, f"{file_name_without_ext}.json")
            
            logger.info(f"Local audio path: {audio_path}")
            logger.info(f"S3 metadata path: {s3_metadata_path}")
            logger.info(f"Local metadata path: {metadata_path}")
            
            try:
                logger.info("Starting S3 downloads...")
                s3_client.download_file(bucket_name, s3_audio_path, audio_path)
                logger.info("Successfully downloaded audio file from S3")
                s3_client.download_file(bucket_name, s3_metadata_path, metadata_path)
                logger.info("Successfully downloaded metadata file from S3")
                final_audio_path.append(audio_path)
                final_metadata_path.append(metadata_path)
            except Exception as e:
                logger.error(f"Failed to download files from S3: {str(e)}")
                logger.error(f"S3 bucket: {bucket_name}")
                logger.error(f"Audio path: {s3_audio_path}")
                logger.error(f"Metadata path: {s3_metadata_path}")
                return False
        
        logger.info("Starting diarization processing (GPT pipeline)...")
        try:
            processor = DiarizationProcessor(message=message, audio_path=final_audio_path, metadata_path=final_metadata_path, logger=logger)
            logger.info("Successfully initialized DiarizationProcessor (GPT)")
            
            data = processor.process_diarization()
            logger.info("Successfully completed diarization processing")
            logger.info(f"Diarization data: {json.dumps(data, indent=2)}")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            logger.error(f"Failed during diarization processing: {str(e)}")
            logger.error(f"Audio path: {audio_path}")
            logger.error(f"Metadata path: {metadata_path}")
            return False

        logger.info("Initializing database connection...")
        # Create tables if they don't exist
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        engine.dispose()
        logger.info("Database tables verified/created successfully")

        # Process database operations
        logger.info("Starting database operations...")
        with get_db_session() as session:
            try:
                logger.info("Creating new CallRecord...")
                # Create new CallRecord instance with data from diarization
                for i in data:
                    call_record = CallRecord(**i)
                    session.add(call_record)
                session.commit()
                logger.info("Successfully created and committed CallRecord")

                logger.info("Updating scheduler status to 'completed'...")
                # Update scheduler table
                session.execute(
                    text("UPDATE scheduler SET call_status = 'completed' WHERE call_id = :call_id"),
                    {"call_id": data[0]["call_id"]}
                )
                session.commit()
                logger.info(f"Successfully updated scheduler call_status for call_id: {call_id}")
                return True

            except SQLAlchemyError as e:
                logger.error("Database operation failed")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                
                if session:
                    logger.info("Rolling back database session...")
                    session.rollback()
                    logger.info("Database session rolled back successfully")
                
                # Update scheduler call_status to failed
                try:
                    logger.info("Attempting to update scheduler call_status to 'failed'...")
                    if message.get('attempt',0) >=5 :
                        session.execute(
                            text("UPDATE scheduler SET call_status = 'failed' WHERE call_id = :call_id"),
                            {"call_id": call_id}
                        )
                    session.commit()
                    logger.info("Successfully updated scheduler status to 'failed'")
                except Exception as db_error:
                    logger.error(f"Failed to update scheduler call_status: {str(db_error)}")
                    logger.error(f"Error type: {type(db_error).__name__}")
                
                return False
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error("="*50)
        logger.error("Unexpected error in process_call")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*50)
        return False
        
    finally:
        logger.info("Starting cleanup process...")
        # Clean up downloaded files
        if final_audio_path:
            try:
                for i in final_audio_path:
                    if os.path.exists(i):
                        os.remove(i)
                logger.info(f"Successfully cleaned up audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to remove audio file: {str(e)}")
                logger.error(f"Audio path: {audio_path}")
                
        if final_metadata_path:
            try:
                for i in final_metadata_path:
                    if os.path.exists(i):
                        os.remove(i)
                logger.info(f"Successfully cleaned up metadata file: {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to remove metadata file: {str(e)}")
                logger.error(f"Metadata path: {metadata_path}")
        
        logger.info("Cleanup process completed")
        logger.info("="*50)

if __name__ == "__main__":
    # Example usage
    callid = "V71BD82QQ56NBFAPR5SRDMCDR8019F3B"
    logger = create_logger(f"logs/{callid}.log")
    message = {
        "call_id": "31162886823203616",
        "call_divisions": [
            {
                "ucid": "31162886855553918",
                "s3_audio_path": "cleaned-audio-recordings-ucid/31162886855553918.mp3",
                "level": "L1"
            }
        ]
    }
    result = process_call(message, logger)
    print(f"Processing completed with status: {result}")
