import os
import csv
import re
from types import SimpleNamespace
from datetime import datetime
import json
import random
import time
import openai
import httpx
from config import *

base_name = os.path.basename(__file__)

def get_last_line(file):
    '''Return the last number in a csv file (used to determine database size)'''
    with open(file, "r") as f:
        return sum(1 for _ in f)

def clear_data():
    '''Clear saved event data (output and embedding files)'''
    choice = input("Are you sure you want to clear all event data? This will not remove saved computation data, however, eventIDs will be reset. (y/n) -> ")
    if choice.lower() == 'y':
        # Remove EVENTS file
        if os.path.exists(EVENTS):
            os.remove(EVENTS)
            print(f"Removed {EVENTS}")
            log(base_name, f"{EVENTS} file removed by user")
        else:
            print(f"{EVENTS} does not exist") # <- file not found
            log(base_name, f"{EVENTS} was attempted to be removed by user but did not exist")
        
        # Remove VECTORS file
        if os.path.exists(VECTORS):
            os.remove(VECTORS)
            print(f"Removed {VECTORS}")
            log(base_name, f"{VECTORS} file removed by user")
        else:
            print(f"{VECTORS} does not exist") # <- file not found
            log(base_name, f"{VECTORS} was attempted to be removed by user but did not exist")

        print()
        return True
    else:
        print("Canceled")
        log(base_name, "User cancelled data removal")
        print()
        return False

def get_time():
    '''Return current time in YYYY-mm-dd HH:MM:SS'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(base, *messages):
    '''Log timestamped message attached to source file'''
    for message in messages:
        with open("log.jsonl", "a") as l:
            record = {
                "time": get_time(),
                "source": base,
                "message": message
            }
            json.dump(record, l)
            l.write("\n")

def get_next_identifier(filename: str = "events.csv") -> str:
    '''
    Returns the next unused eventID from existing rows in EVENTS
    '''
    max_n = 0

    # Start at evt001 if file dne
    if not os.path.isfile(filename):
        return "evt001"
    
    # Capture only number from eventID
    pattern = re.compile(r"^evt(\d+)$")

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ident = (row.get("identifier") or "").strip()
            m = pattern.match(ident)
            if m:
                max_n = max(max_n, int(m.group(1)))

    return f"evt{max_n + 1:03d}"

def read_csv_line_to_object(filename: str, row_number: int = 1) -> SimpleNamespace:
    '''
    Read one data row from a CSV and return it as an object.

    row_number is 1-based and counts only data rows (not the header).
    '''
    if row_number < 1:
        raise ValueError("row_number must be >= 1")

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [h.lstrip("\ufeff") for h in reader.fieldnames]

        for index, row in enumerate(reader, start=1):
            if index == row_number:
                return SimpleNamespace(**row)

    raise ValueError(f"Row {row_number} not found in {filename}")

def get_client_type():
    '''Return the client type as a string, if not gemini or claude, assumes model is compatible with openAI module'''
    m = (MODEL_NAME or "").lower()
    b = (API_BASE_URL or "").lower()

    if "anthropic" in b or m.startswith("claude"):
        return "anthropic"
    if "googleapis.com" in b or "generativelanguate" in b or m.startswith("gemini"):
        return "gemini"
    return "openai_compatible"

def send_prompt(input: str, event_id: str, output_id: int) -> str:
    '''Send prompt to client and return raw output'''
    client_type = get_client_type()
    
    # Claude
    if client_type == "anthropic":
        # Call claude
        pass

    # Gemini
    elif client_type == "gemini":
        # Call gemini
        pass

    # GPT / OpenAI Compatible
    else:
        # Call OpenAI
        output = prompt_openai(input, event_id, output_id)

    return output

def prompt_openai(prompt: str, event_id: str, output_id: int, max_attempts: int = 5) -> str:
    '''
    Send one prompt with retry/backoff
    '''
    for attempt in range(1, max_attempts + 1):
        try:
            response = CLIENT.responses.create(
                model=MODEL_NAME,
                input=prompt
            )
            output =  " ".join(response.output_text.split())
            return output
        except Exception as e:
            error_text = str(e)
            if attempt >= max_attempts:
                log(
                    base_name,
                    f"{event_id} output-{output_id} exhausted retries after {max_attempts} attempts. "
                    f"Last error: {error_text}"
                )
                raise RuntimeError(f"Prompt failed after {max_attempts} attempts: {error_text}") from e

            backoff = min(60.0, 2 ** min(attempt, 6)) + random.uniform(0.0, 1.0)
            log(
                base_name,
                f"{event_id} output-{output_id} attempt-{attempt} failed: {error_text}. "
                f"Retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)

    raise RuntimeError(f"Prompt failed after {max_attempts} attempts")
