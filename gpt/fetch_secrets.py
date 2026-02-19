#!/usr/bin/env python3
"""
GPT Pipeline Secrets Fetcher - Fetches secrets from AWS Secrets Manager.
Cleaned version of src/fetch_secrets.py without GCP dependencies.
"""

import json
import os
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_secret(secret_name, region_name="ap-south-1"):
    """Retrieve a secret from AWS Secrets Manager."""
    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        raise
    else:
        if 'SecretString' in get_secret_value_response:
            secret = json.loads(get_secret_value_response['SecretString'])
            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret
        else:
            logger.error(f"Secret {secret_name} not found or invalid")
            raise ValueError(f"Secret {secret_name} not found or invalid")

def create_client_properties(kafka_config):
    """Create client.properties file from Kafka configuration."""
    client_properties_content = f"""# Required connection configs for Kafka producer, consumer, and admin
bootstrap.servers={kafka_config.get('bootstrap.servers', '')}
security.protocol={kafka_config.get('security.protocol', 'SASL_SSL')}
sasl.mechanisms={kafka_config.get('sasl.mechanisms', 'PLAIN')}
sasl.username={kafka_config.get('sasl.username', '')}
sasl.password={kafka_config.get('sasl.password', '')}

# Best practice for higher availability in librdkafka clients prior to 1.7
session.timeout.ms={kafka_config.get('session_timeout_ms', '45000')}

client.id={kafka_config.get('client_id', 'ccloud-python-client')}
"""
    # Use current directory instead of /app/ to be more flexible
    with open('client.properties', 'w') as f:
        f.write(client_properties_content)
    
    logger.info("Created client.properties file")

def set_environment_variables(env_config):
    """Set environment variables from configuration."""
    for key, value in env_config.items():
        os.environ[key] = str(value)
    
    logger.info(f"Set {len(env_config)} environment variables")
