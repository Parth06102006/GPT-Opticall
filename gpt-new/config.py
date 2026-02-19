from dotenv import load_dotenv
import boto3
import os
import sys
import pandas as pd

# Load environment variables from .env file
load_dotenv()

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Required environment variable {var_name} is not set")
    return value

# AWS Region name
region_name = get_required_env_var('region_name')
ACCESS_KEY = get_required_env_var('ACCESS_KEY')
SECRET_KEY = get_required_env_var('SECRET_KEY')

# Bucket name
bucket_name = get_required_env_var('bucket_name')

opticall_dataframes_api = get_required_env_var('opticall_dataframes_api')
login_api = get_required_env_var('login_api')
user_name = get_required_env_var('user_name')
user_password = get_required_env_var('user_password')

login_data = {
    "username": user_name,
    "password": user_password
}

# Initialize AWS clients
s3_client = boto3.client('s3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=region_name
)
s3 = boto3.resource('s3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=region_name
)

audio_folder = "audio"
os.makedirs(audio_folder, exist_ok=True)
metadata_folder = "metadata"
os.makedirs(metadata_folder, exist_ok=True)

os.makedirs("logs", exist_ok=True)

# Kafka configuration
opticall_audio_topic = get_required_env_var('opticall_audio_topic')
opticall_audio_error_topic = get_required_env_var('opticall_audio_error_topic')
opticall_audio_group = get_required_env_var('opticall_audio_group')
opticall_batch_topic = get_required_env_var('opticall_batch_topic')
opticall_batch_error_topic = get_required_env_var('opticall_batch_error_topic')
opticall_batch_group = get_required_env_var('opticall_batch_group')

def read_config():
    # reads the client configuration from client.properties
    config = {}
    try:
        with open("client.properties") as fh:
            for line in fh:
                line = line.strip()
                if len(line) != 0 and line[0] != "#":
                    parameter, value = line.strip().split('=', 1)
                    config[parameter] = value.strip()
        return config
    except FileNotFoundError:
        print("Warning: client.properties file not found. Kafka may require local config.")
        return {}
    except Exception as e:
        print(f"Error reading client.properties: {str(e)}")
        return {}

config = read_config()

# Database and Paths
DATABASE_URL = get_required_env_var('DATABASE_URL')

# Path to prompts directory relative to this file
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

SYSTEM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "system.txt")
USER_PROMPT_AGENT_PATH = os.path.join(PROMPTS_DIR, "agent.txt")
USER_PROMPT_AGENT_CUSTOMER_PATH = os.path.join(PROMPTS_DIR, "agent_customer.txt")
USER_PROMPT_COMMON_HEADERS_PATH = os.path.join(PROMPTS_DIR, "common_headers.txt")
USER_PROMPT_AGENT_METRICS_AUDIO_PATH = os.path.join(PROMPTS_DIR, "agent_metrics_audio.txt")
USER_PROMPT_CALL_METRICS_TRANSCRIPT_PATH = os.path.join(PROMPTS_DIR, "call_metrics_transcript.txt")
AGENT_SCRIPT_DISH_PATH = os.path.join(PROMPTS_DIR, "dish_script.csv")
AGENT_SCRIPT_D2H_PATH = os.path.join(PROMPTS_DIR, "d2h_script.csv")

# Load scripts
dish_script = pd.read_csv(AGENT_SCRIPT_DISH_PATH)
d2h_script = pd.read_csv(AGENT_SCRIPT_D2H_PATH)

NOUN_CORPUS_PATH = os.path.join(PROMPTS_DIR, "transcript_corpus.xlsx")
noun_corpus_temp = pd.read_excel(NOUN_CORPUS_PATH)
noun_corpus = noun_corpus_temp.iloc[:100].to_string(index=False)

