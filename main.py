
import os
from config import *
import random
import csv
from helper import get_time, log, get_next_identifier, read_csv_line_to_object, clear_data, get_last_line, send_prompt
import json

base_name = os.path.basename(__file__)

log(base_name, "Beginning encoding setup")
encoding_model = get_transformer_model()
log(base_name, "Encoding setup complete")

###############################################################################################
## Start
def main():
    # Ask user which database
    db_list = list(DATABASES.items())
    choice = None
    while True:
        # Display database menu
        print("Select article source")
        while True: # loops only if 'clr' or 'format' entered
            for i, (db_key, db_info) in enumerate(db_list, start=1):
                print(f"{i}. {db_info['alias']}")

            choice = input("-> ").strip()

            # Remove data
            if choice == 'clr':
                clear_data()
                continue
            # Remove data and erase log (test use only) - not mentioned in README
            elif choice == 'format':
                if clear_data():
                    if os.path.exists('log.jsonl'):
                        os.remove('log.jsonl')
                        log(base_name, "Format occured")
            break

        # Verify numeric
        if not choice.isdigit():
            print("You must enter a numerical value")
            continue

        # Verify in bound
        choice_num = int(choice)
        if not (1 <= choice_num <= len(db_list)):
            print(f"You must enter a number between 1 and {len(db_list)}")
            continue

        # Match selected database
        selected_key, selected = db_list[choice_num - 1]
        database = selected['filename'] # csv file name
        database_id = selected['alias'] # label
        break

    # Ask user how many simulations
    while True:
        try:
            sim_runs = int(input("Select # of simulations -> "))
        except ValueError:
            print("You must enter a valid integer")
        break

    ### BEGIN SIMULATION ##############################################################
    upper_bound = get_last_line(database) - 1 # <- Get total num of articles in database

    # EVENTS file header
    event_header = [
        "identifier", "outuput_id", "database", "timestamp", "topic",
        "phase", "article_id", "order", "persona_id", "model_name", "output"
    ]

    def append_event_row(row):
        # Write header in EVENTS file if it is empty or doesn't exist
        needs_header = (not os.path.exists(EVENTS)) or (os.path.getsize(EVENTS) == 0)

        # Append row to EVENTS file
        with open(EVENTS, "a", newline="", encoding="utf-8") as events_file:
            writer = csv.writer(events_file)
            if needs_header:
                writer.writerow(event_header)
            writer.writerow(row)

    # Event
    for e in range(sim_runs):
        event_id = get_next_identifier(EVENTS)

        print()
        print(f"Starting simulation event \033[32m{event_id} \033[0m| Session simulation \033[36m{e+1}\033[0m/{sim_runs}")

        # Log sim start
        log(base_name, f"{event_id} sim started from {database_id} | {e+1}/{sim_runs} in session")

        output_id = 1

        ##### BEFORE #####
        before_articles = []
        before_row_ids = set() # <- csv row ids (used to prevent duplicate selection in after phase)
        used_topics = set()

        # Select 4 random articles from each topic
        while len(before_articles) < 4:
            r = random.randint(1, upper_bound)
            if r in before_row_ids:
                continue

            rand_article = read_csv_line_to_object(database, r)

            if rand_article.topic in used_topics:
                continue

            before_articles.append(rand_article)
            before_row_ids.add(r)
            used_topics.add(rand_article.topic)

        # For each of 4 articles
        for before_article_idx, article in enumerate(before_articles):

            print(f"Selected | {database_id} - {article.article_id}")

            # For each of 4 personas
            for p_idx, persona in enumerate(PERSONAS):

                # Build prompt
                prompt = (
                        f"{PREFACE}\n\n"
                        f"PERSONA:\n{persona}\n\n"
                        f"ARTICLE:\n{article.text}"
                    )

                failed = False
                try:
                    output = send_prompt(prompt, event_id, output_id)
                except Exception as e:
                    print(f"{type(e).__name__} exception logged: disregard {event_id} output {output_id} vector, do not use it for any calculations.")
                    log(base_name, f"{type(e).__name__}: {e}")
                    output = f"{type(e).__name__}[ERROR]"
                    failed = True

                # Terms of service violation
                if output in TOS_MESSAGES:
                    print (f"\033[31mModel terms of service violated, \033[33mOutput {event_id}--{output_id} will not be vectorized.\033[0mIf this issue persists, consider changing prompt.")
                    log(base_name, f"TOS Violation {event_id}-{output_id}")
                    failed = True

                # Save event data to EVENTS
                append_event_row([
                    event_id,
                    output_id,
                    database,
                    get_time(),
                    article.topic,
                    "before",
                    article.article_id,
                    f"{before_article_idx + 1}",
                    f"p{p_idx + 1}",
                    MODEL_NAME,
                    output
                ])

                # Embed response if valid
                if not failed:
                    # Generate and save vectorized output only for successful outputs.
                    output_vector = encoding_model.encode(output).astype("float32").tolist()
                    with open(VECTORS, "a") as f:
                            record = {
                                'id': event_id,
                                'o': output_id, # output num
                                's': "BEFORE", # state
                                'a': article.article_id, # articleID
                                'db': article.database, # database
                                'p': p_idx+1, # persona
                                't': article.topic, # topic
                                'vector': output_vector # vectorized output
                            }
                            json.dump(record, f, separators=(",", ":"))
                            f.write("\n")

                    # Log successful response
                    log(base_name, f"{event_id}-{output_id} completed successfully")

                output_id += 1

        ##### AFTER #####
        print("Beginning second phase")
        log(base_name, f"phase 2 of {event_id} starting")

        after_articles = []
        used_topics = set()
        after_row_ids = set()

        # Select 4 random articles from each topic
        while len(after_articles) < 4:
            r = random.randint(1, upper_bound)

            # Prevent overlap with before articles
            if r in before_row_ids:
                continue

            if r in after_row_ids:
                continue

            rand_article = read_csv_line_to_object(database, r)

            # Prevent duplicate topics (within AFTER)
            if rand_article.topic in used_topics:
                continue

            after_articles.append(rand_article)
            after_row_ids.add(r)
            used_topics.add(rand_article.topic)

        # For each of 4 articles
        for after_article_idx, article in enumerate(after_articles):

            print(f"Selected | {database_id} - {article.article_id}")

            # For each of 4 personas
            for p_idx, persona in enumerate(PERSONAS):

                # Build prompt
                prompt = (
                        f"{PREFACE}\n\n"
                        f"{INTERVENTION_PREFACE}\n\n{INTERVENTION}\n\n" # Insert intervention article
                        f"PERSONA:\n{persona}\n\n"
                        f"ARTICLE:\n{article.text}"
                    )
                failed = False
                try:
                    output = send_prompt(prompt, event_id, output_id)
                except Exception as e:
                    print(f"{type(e).__name__} exception logged: disregard {event_id} output {output_id} vector, do not use it for any calculations.")
                    log(base_name, f"{type(e).__name__}: {e}")
                    output = f"{type(e).__name__}[ERROR]"
                    failed = True


                # Terms of service violation
                if output in TOS_MESSAGES:
                    print (f"\033[31mModel terms of service violated, \033[33mOutput {event_id}--{output_id} will not be vectorized.\033[0mIf this issue persists, consider changing prompt.")
                    log(base_name, f"TOS Violation {event_id}-{output_id} Model: {MODEL_NAME}")
                    failed = True

                # Save event
                append_event_row([
                    event_id,
                    output_id,
                    database,
                    get_time(),
                    article.topic,
                    "after",
                    article.article_id,
                    f"{after_article_idx + 1}",
                    f"p{p_idx + 1}",
                    MODEL_NAME,
                    output
                ])

                # Embed response if valid
                if not failed:
                    # Generate and save vector only for successful outputs.
                    output_vector = encoding_model.encode(output).astype("float32").tolist()
                    with open(VECTORS, "a") as f:
                            record = {
                                'id': event_id,
                                'o': output_id,
                                's': "AFTER", # state
                                'a': article.article_id, # articleID
                                'db': article.database, # database
                                'p': p_idx+1, # persona
                                't': article.topic, # article topic
                                'vector': output_vector # vectorized output
                            }
                            json.dump(record, f, separators=(",", ":"))
                            f.write("\n")

                    # Log successful response
                    log(base_name, f"{event_id}-{output_id} completed successfully")

                output_id += 1

        # Log sim completion for each event
        message = f"{event_id} sim completed from {database_id}"
        log(base_name, message)

    ##### End session #####
    print(f"{sim_runs} Simulation(s) complete, all simulation data saved to saved to {EVENTS}")

## Run main
if (__name__ == "__main__"):
    try:
        log(base_name, f"Session started")
        main()
        log(base_name, "Session completed")
    except KeyboardInterrupt as k:
        log(base_name, f"{type(k).__name__}: session terminated")
        print("Terminating session")
    except Exception as e:
        log(base_name, f"{type(e)}: {e}")
        print(f"\033[33mFatal {type(e).__name__} occured, check log.jsonl for more information. \033[31mTerminating session.\033[0m")
