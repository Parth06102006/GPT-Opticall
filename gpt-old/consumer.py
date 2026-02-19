"""
GPT Pipeline Consumer - Kafka consumer for real-time audio processing.

Same Kafka + S3 flow as the original consumer.py, but:
  - No GCP credential fetching (uses OpenAI instead)
  - Imports processor from gpt.processor (OpenAI-based)
"""
import sys
import os

# Ensure project root is in sys.path for src.* imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gpt.kafkaConnector import kafkaConnector
import time
from gpt.logger import create_logger
import boto3
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from gpt.fetch_secrets import get_secret, create_client_properties, set_environment_variables
from sqlalchemy import text

def initialize_secrets():
    """
    Initialize secrets from AWS Secrets Manager.
    Fetches Kafka config and environment variables.
    NOTE: No GCP credentials needed — this pipeline uses OpenAI API keys via env vars.
    """
    try:
        # Get AWS region from environment or use default
        aws_region = os.getenv('AWS_REGION', 'ap-south-1')
        
        # Secret names - these should be configured in your deployment
        secrets_config = {
            'env_secret_name': os.getenv('ENV_SECRET_NAME', 'opticall/env-variables'),
            'kafka_secret_name': os.getenv('KAFKA_SECRET_NAME', 'opticall/kafka-config'),
        }
        
        print("Starting secrets initialization (GPT pipeline - no GCP)...")
        
        # Fetch Kafka configuration
        print(f"Fetching Kafka configuration from {secrets_config['kafka_secret_name']}")
        kafka_config = get_secret(secrets_config['kafka_secret_name'], aws_region)
        create_client_properties(kafka_config)

        # Fetch environment variables (includes OPENAI_API_KEY)
        print(f"Fetching environment variables from {secrets_config['env_secret_name']}")
        env_config = get_secret(secrets_config['env_secret_name'], aws_region)
        set_environment_variables(env_config)
        
        print("All secrets have been successfully fetched and configured (GPT pipeline)")
        return True
        
    except ImportError:
        print("Warning: fetch_secrets module not found. Running without secrets management.")
        return False
    except Exception as e:
        print(f"Error initializing secrets: {e}")
        return False

# Initialize secrets before setting up the main logger
secrets_initialized = initialize_secrets()

from gpt.config import opticall_audio_topic, opticall_audio_error_topic, opticall_audio_group
from gpt.processor import get_db_session, process_call

# Initialize main logger for the consumer
main_logger = create_logger("logs/gpt_consumer_main.log")

def read_kafka_config():
    """
    Read Kafka configuration from client.properties file.
    
    Returns:
        dict: Dictionary containing Kafka configuration
    """
    main_logger.info("Reading Kafka configuration from client.properties")
    config = {}
    try:
        with open('client.properties', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        main_logger.info("Successfully read Kafka configuration")
        print(f"Kafka config: {config}")
        return config
    except Exception as e:
        main_logger.error(f"Failed to read Kafka configuration: {str(e)}")
        raise

def process_message(message):
    try:
        main_logger.info(f"Received message for processing: {message}")
        document_id = message.get("call_id", "unknown")
        main_logger.info(f"Processing document ID: {document_id}")
        
        log_file_name = f"logs/{document_id}.log"
        logger = None
        
        try:
            logger = create_logger(log_file_name)
            logger.info(f"Started the processing of {document_id} file (GPT pipeline).")
            start_time = time.time()

            status = process_call(message, logger)
            if not status:
                logger.info(f"File {document_id} processing failed, will be sent to error topic.")
                return False
            else:
                logger.info(f"Successfully processed document {document_id}")

            logger.info(f"Ended the processing of {document_id} file.")
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Total time taken to process the file {document_id} is {processing_time} seconds.")
            main_logger.info(f"Completed processing document {document_id} in {processing_time} seconds")
            return True
            
        except Exception as e:
            main_logger.error(f"Error processing message for document {document_id}: {str(e)}")
            if logger:
                logger.error(f"Error processing message: {str(e)}")
            return False
        finally:
            # Clean up logger handlers to close file handles
            if logger:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
    except Exception as e:
        main_logger.error(f"Error in process_message: {str(e)}")
        return False

def main():
    main_logger.info("Starting GPT Kafka consumer service")
    
    # Read Kafka configuration from properties file
    kafka_config = read_kafka_config()
    
    # Initialize Kafka connector with configuration
    main_logger.info(f"Initializing Kafka connector with topic: {opticall_audio_topic}")
    
    # Create a single Kafka connector instance outside the loop
    client = kafkaConnector(
        main_topic=opticall_audio_topic,
        error_topic=opticall_audio_error_topic,
        group_id=opticall_audio_group,
        config=kafka_config
    )
    
    main_logger.info("Entering main consumer loop (GPT pipeline)")
    try:
        while True:
            try:
                status, messages = client.get_messages(mode="main")
                if not status:
                    main_logger.debug("No messages received, waiting for next poll")
                    time.sleep(client.repoll_time)
                    continue

                main_logger.info(f"Received {len(messages)} messages for processing")

                # Create thread pool with a context manager
                with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 8))) as executor:
                    main_logger.info(f"Starting processing of {len(messages)} messages with thread pool")
                    # Submit each message for processing
                    futures = [executor.submit(process_message, message) for message in messages]
                    
                    # Process results and handle failures
                    failed_messages = []
                    for message, future in zip(messages, futures):
                        try:
                            # Add 2-minute timeout for each thread
                            success = future.result(timeout=120)
                            if not success:
                                # Increment attempt for normal failure
                                if "attempt" in message.keys():
                                    message["attempt"] += 1
                                else:
                                    message["attempt"] = 1
                                failed_messages.append(message)
                        except TimeoutError:
                            document_id = message.get("call_id", "unknown")
                            main_logger.error(f"Thread processing document {document_id} timed out after 2 minutes")
                            # Increment attempt for timeout before appending
                            if "attempt" in message.keys():
                                message["attempt"] += 1
                            else:
                                message["attempt"] = 1
                            # Cancel the future to prevent resource leaks
                            future.cancel()
                            failed_messages.append(message)
                        except Exception as e:
                            document_id = message.get("call_id", "unknown")
                            main_logger.error(f"Error processing message for {document_id}: {str(e)}")
                            # Increment attempt for exception before appending
                            if "attempt" in message.keys():
                                message["attempt"] += 1
                            else:
                                message["attempt"] = 1
                            # Cancel the future to prevent resource leaks
                            future.cancel()
                            failed_messages.append(message)

                    # Wait for all remaining futures to complete or be cancelled
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    
                    main_logger.info("Completed processing all messages in current batch")
                    
                    # If we have failed messages, send them to error topic
                    if failed_messages:
                        main_logger.info(f"Sending {len(failed_messages)} failed messages to error topic")
                        for failed_message in failed_messages:
                            if failed_message.get("attempt",0) < 2:
                                try:
                                    client.send_error_message(failed_message)
                                except Exception as e:
                                    main_logger.error(f"Failed to send message to error topic: {str(e)}")
                            else:
                                with get_db_session() as session:
                                    query = text("UPDATE scheduler SET call_status = 'failed' WHERE call_id = :call_id")
                                    session.execute(query, {"call_id": failed_message.get('call_id')})
                                    session.commit()
                                main_logger.error(f"Skipping message {failed_message.get('call_id')} as it has reached maximum attempts")
                    
            except Exception as e:
                main_logger.error(f"Error in main consumer loop: {str(e)}")
                time.sleep(client.repoll_time)  # Add delay before retrying
                
    except KeyboardInterrupt:
        main_logger.info("Received keyboard interrupt, shutting down gracefully")
    except Exception as e:
        main_logger.error(f"Fatal error in consumer: {str(e)}")
    finally:
        main_logger.info("Closing Kafka consumer")
        try:
            if client.consumer:
                client.consumer.close()
        except Exception as e:
            main_logger.error(f"Error closing consumer: {str(e)}")

if __name__ == "__main__":
    main()
