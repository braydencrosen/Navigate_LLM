import os
import httpx
import openai
from sentence_transformers import SentenceTransformer
import openai
import httpx

#### MODEL VARS ####
API_KEY = os.environ.get("API_KEY") # <- retrieved from environment, see README.md
MODEL_NAME = "Insert model name here"
API_BASE_URL = "Insert API base URL here"
CLIENT = openai.OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    max_retries=10, # <- Automatic retry on transient failures
)

#### SENTENCE TRANSFORMERS ####

# Silence HF/Transformers output before import
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import logging as tlog
tlog.set_verbosity_error()
_TRANSFORMER_MODEL = None
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

def get_transformer_model():
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        _TRANSFORMER_MODEL = SentenceTransformer(EMBEDDING)
    return _TRANSFORMER_MODEL


#### FILE MANAGEMENT ####
EVENTS = "events.csv"
file_exists = os.path.isfile(EVENTS)

# Database management, more could be added
# alias is the display name for the database, filename is used for the path
DATABASES = {
    "db1": {"alias": "db1", "filename": "database1.csv"},
    "db2": {"alias": "db2", "filename": "database2.csv"},
}
VECTORS = "event_vectors.jsonl" # <- Vector save file
REPORT_FOLDER = "Computation_Data" # <- Computation save file

#### PROMPT VARS ####
TOS_MESSAGES = [
    "Insert any known terms of service violation messages here" # <- When this text is detected, output will automatically not be saved or vectorized
]

# Intervention preface (this sends before each intervention injection in the AFTER stage)
INTERVENTION_PREFACE = "Do not respond to this article."

INTERVENTION = "This is the text that will be inserted in the after phase in the simulation."

# LLM Personas (iterated through in simulation)
PERSONAS = ["Insert persona 1 here",
            "Insert persona 2 here",
            "Insert persona 3 here",
            "Insert persona 4 here"
            ]

# Prompt preface (this sends before each prompt no matter the stage)
PREFACE = (
            "This is the text used to give the LLM context on the experiment"
)
