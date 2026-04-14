from helper import log, get_time, VECTORS, EVENTS, REPORT_FOLDER, DATABASES
import os
import json
import csv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from itertools import combinations
from pathlib import Path
from textwrap import wrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from vizualize import generate_visualizations
from config import EMBEDDING

base_name = os.path.basename(__file__)

##### UTILITY #####
def save_rows_to_dict():
    '''Read each line from VECTORS file and return it as a list of dicts'''

    if not os.path.exists(VECTORS):
        log(base_name, f"{VECTORS} file was not found")
        print(f"\033[31mError\033[0m fetching \033[33m{VECTORS}\033[0m please check that it exists and is in the project directory.")
        raise FileNotFoundError
    
    rows = [] # <- List to hold dicts

    try:
        with open(VECTORS, "r", encoding="utf-8") as f:
            
            lines = {}
            for line in f:
                # Clear whitespace trail
                line = line.strip()

                # Skip empty liens
                if not line:
                    continue

                row = json.loads(line)

                # Create dict
                key = (
                    str(row["id"]),
                    int(row["o"]),
                    str(row["s"]).upper(),
                    str(row["a"]),
                    str(row["db"]),
                    str(row["p"]),
                    str(row["t"]),
                )
                lines[key] = row
                rows.append(row)
    except Exception as e:
        log(base_name, f"{type(e).__name__}: {e}")
        print(f"\033[31mError\033[0m {VECTORS} file structure may be invalid. Please check log.jsonl for more information")

    return rows

def get_event_selection(vector_data, cross_choice=None, cross_compute=False):
    '''List available events in VECTORS and prompt the user to select one
    
    if cross_compute is true, console will display "Select vector {cross_choice}",
    which can be used when multiple events need to be selected.'''
    events = []
    seen = [] # <- Variable to store "seen" events so they are only added once

    # Find all databases in vector file
    for row in vector_data:
        event_id = row['id']
        db = row['db']
        if event_id not in seen:
            seen.append(event_id)
            events.append((event_id, db))

    if cross_compute:
        # Display events (A/B)
        print(f"Select event {cross_choice}")

    # Display events
    for i, (event_id, db) in enumerate(events, start=1):
        print(f"{i}. {event_id} ({db})")

    while True:
        try:
            choice = input("-> ")
            choice = int(choice)

        except ValueError:
            print("You must enter a valid integer")
            continue

        if not (1 <= choice <= len(events)):
            print(f"You must enter a number between 1 and {len(events)}")
            continue

        chosen_event = events[choice-1][0]
        return chosen_event

def get_all_event_ids(vector_data):
    '''Return all unique event IDs in VECTORS file'''
    event_ids = []
    seen = set()
    for row in vector_data:
        event_id = row["id"]
        if event_id in seen:
            continue
        seen.add(event_id)
        event_ids.append(event_id)
    return event_ids
    
def get_topics(data):
    '''Return list of topics found in given VECTORS data (can be partition or in full)'''
    topics = []
    seen = []
    for row in data:
        topic = row['t']
        if topic not in seen:
            seen.append(topic)
            topics.append(topic)

    return topics

def get_personas(data):
    '''Return list of personas found in given VECTORS data (can be partition or in full)'''
    personas = []
    seen = []
    for row in data:
        persona = row['p']
        if persona not in seen:
            seen.append(persona)
            personas.append(persona)

    return personas

def select_event_vectors(vector_data, eventID):
    '''Return all vectors in one event'''
    event_vectors = []

    for row in vector_data:
        if row['id'] == eventID:
            event_vectors.append(row)

    return event_vectors

def get_database(db):
    '''Return database filename based on db identifier stored in VECTORS'''

    db_value = str(db).strip().lower()

    for db_key, db_info in DATABASES.items():
        key_value = str(db_key).strip().lower()
        alias_value = str(db_info.get("alias", "")).strip().lower()
        filename_value = str(db_info.get("filename", "")).strip().lower()
        alias_prefix = alias_value.split("(", 1)[0].strip()

        if db_value in {key_value, alias_value, alias_prefix, filename_value}:
            return db_info["filename"]

    log(base_name, f"Error resolving database using get_database(), db={db}")
    print("An error occured determining database, check log.jsonl for more information.")
    quit()

def get_vector(event_vectors, topic, persona, stage):
    '''Return a single vector associated with given event, topic, persona, and state'''
    return next(
        (
            row["vector"] for row in event_vectors
            if row["t"] == topic and row["p"] == persona and row["s"] == stage
        ),
        None
    )

def save_report(report_text, viz_results=None, viz_mode="event", viz_title="Computation", filename=None, clear_after_save=True):
    '''Save report in designated folder as .pdf or .txt'''
    # Get file name and type
    if filename is None:
        while True:

            # Get filename and extract extension
            filename = input("Enter file name ending in .txt or .pdf -> ")
            extension = filename[-4:]

            # Missing file extension
            if extension not in [".txt", ".pdf"]:
                log(base_name, f"User attempted save with invalid or missing file extension {filename}")
                print("Invalid or missing file extension, try again.")
                continue

            # File exists already
            elif (Path(REPORT_FOLDER) / filename).exists():
                print("This file already exists, please choose another name.")
                continue
            break
    else:
        extension = filename[-4:]
        if extension not in [".txt", ".pdf"]:
            log(base_name, f"Invalid file extension for save_report filename={filename}")
            print("Invalid or missing file extension.")
            return None
        if (Path(REPORT_FOLDER) / filename).exists(): # <- Skip over duplicate filenames to prevent duplicate saving
            log(base_name, f"Report already exists, skipping save {filename}")
            print(f"Skipping existing file: {filename}")
            return None

    # Set output file path
    output_file = Path(REPORT_FOLDER) / filename

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate plots only when save is confirmed
    plot_paths = []
    if viz_results is not None:
        plot_paths = generate_visualizations(viz_results, str(output_file), mode=viz_mode, title_prefix=viz_title)

    # .txt file save
    if extension == ".txt":
        log(base_name, f"Saving text report to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
    # .pdf file save
    if extension == ".pdf":
        log(base_name, f"Saving pdf report to {output_file}")
        c = canvas.Canvas(str(output_file), pagesize=letter)
        _, height = letter
        margin = 50
        y = height - margin
        font_name = "Helvetica"
        font_size = 9
        line_height = 11
        max_chars = 120

        c.setFont(font_name, font_size)

        for paragraph in report_text.splitlines():
            lines = wrap(paragraph, width=max_chars) if paragraph else [""]
            for line in lines:
                if y <= margin:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - margin
                c.drawString(margin, y, line)
                y -= line_height

        # Draw generated plots at the end of the PDF content.
        max_plot_width = letter[0] - (2 * margin)
        max_plot_height = (letter[1] - (2 * margin)) * 0.45
        for plot_path in plot_paths:
            try:
                image = ImageReader(plot_path)
                img_w, img_h = image.getSize()
                scale = min(max_plot_width / img_w, max_plot_height / img_h)
                draw_w = img_w * scale
                draw_h = img_h * scale

                if y - draw_h - line_height <= margin:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - margin

                c.drawImage(plot_path, margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
                y -= draw_h + line_height
            except Exception as e:
                log(base_name, f"Failed to embed plot {plot_path} in PDF: {type(e).__name__}: {e}")

        c.save()

    if clear_after_save:
        clear()
    print(f"Saved {output_file}")
    return str(output_file)

def propmt_save() -> bool:
    '''Determine if user wants to save data, with cancellation confirmation'''
    save = input("Save data? (y/n) -> ").strip().lower() in ["y", "yes"]
    if not save:
        sure_nosave = input("Are you sure you want to \033[33mdelete\033[0m this data? This cannot be undone. (y/n) -> ").strip().lower() in ["y", "yes"]
        if sure_nosave:
            save = False
        else:
            save = True
    return save

def prompt_show() -> bool:
    '''Determine if user wants to display data to terminal'''
    show = input("Output generated, display it? (y/n) -> ").strip().lower() in ["y", "yes"]
    return show

##### DISPLAY #####
def clear():
    '''Clears terminal display'''
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

def clean_filename(value):
    '''Convert value to a filename-safe token'''
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    safe = safe.strip("._")
    return safe or "unknown_event"

def persona_sort_key(value):
    '''Cleanly order personas'''
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))

def build_batch_summary(event_outputs):
    '''Build topic/persona average summary across all event BEFORE vs AFTER results'''
    summary_map = {}
    # Get all database aliases
    db_aliases = [info.get("alias", key) for key, info in DATABASES.items()]

    # Build empty headers
    db_header_cosine = {alias: [] for alias in db_aliases}
    db_header_euclidean = {alias: [] for alias in db_aliases}
    
    
    persona_order = ["1", "2", "3", "4"]
    db_persona_cosine = {db: {p: [] for p in persona_order} for db in db_aliases}
    db_persona_euclidean = {db: {p: [] for p in persona_order} for db in db_aliases}

    for event_output in event_outputs:
        event_id = event_output["event_id"]
        database = event_output.get("database")
        database_text = str(database).strip()
        database_lower = database_text.lower()
        database_alias = database_text if database_text else "unknown_db"

        for db_key, db_info in DATABASES.items():
            alias = str(db_info.get("alias", db_key)).strip()
            alias_lower = alias.lower()
            filename = str(db_info.get("filename", "")).strip()
            filename_lower = filename.lower()
            alias_prefix = alias_lower.split("(", 1)[0].strip()

            if database_lower in {str(db_key).strip().lower(), alias_lower, alias_prefix, filename_lower}:
                database_alias = alias
                break

        rows = event_output["before_after_results"].get(event_id, [])

        for row in rows:
            topic = row.get("topic")
            persona = row.get("persona")
            if topic is None or persona is None:
                continue
            persona_key = str(persona)

            cosine = row.get("cosine")
            euclidean = row.get("euclidean")
            key = (topic, persona)

            if key not in summary_map:
                summary_map[key] = {
                    "topic": topic,
                    "persona": persona,
                    "cosine_values": [],
                    "euclidean_values": [],
                    "db_cosine_values": {},
                    "db_euclidean_values": {},
                }

            if isinstance(cosine, (int, float)):
                summary_map[key]["cosine_values"].append(float(cosine))
                summary_map[key]["db_cosine_values"].setdefault(database_alias, []).append(float(cosine))
                if database_alias in db_header_cosine:
                    db_header_cosine[database_alias].append(float(cosine))
                if database_alias in db_persona_cosine and persona_key in db_persona_cosine[database_alias]:
                    db_persona_cosine[database_alias][persona_key].append(float(cosine))
            if isinstance(euclidean, (int, float)):
                summary_map[key]["euclidean_values"].append(float(euclidean))
                summary_map[key]["db_euclidean_values"].setdefault(database_alias, []).append(float(euclidean))
                if database_alias in db_header_euclidean:
                    db_header_euclidean[database_alias].append(float(euclidean))
                if database_alias in db_persona_euclidean and persona_key in db_persona_euclidean[database_alias]:
                    db_persona_euclidean[database_alias][persona_key].append(float(euclidean))

    summary_rows = []
    for topic, persona in sorted(summary_map.keys(), key=lambda x: (str(x[0]), persona_sort_key(x[1]))):
        values = summary_map[(topic, persona)]
        cosine_values = values["cosine_values"]
        euclidean_values = values["euclidean_values"]
        db_cosine_values = values["db_cosine_values"]
        db_euclidean_values = values["db_euclidean_values"]

        avg_cosine = sum(cosine_values) / len(cosine_values) if cosine_values else None
        avg_euclidean = sum(euclidean_values) / len(euclidean_values) if euclidean_values else None
        samples = max(len(cosine_values), len(euclidean_values))
        db_avg_cosine = {
            db: (sum(db_values) / len(db_values))
            for db, db_values in db_cosine_values.items() if db_values
        }
        db_avg_euclidean = {
            db: (sum(db_values) / len(db_values))
            for db, db_values in db_euclidean_values.items() if db_values
        }
        db_samples = {
            db: max(len(db_cosine_values.get(db, [])), len(db_euclidean_values.get(db, [])))
            for db in set(db_cosine_values) | set(db_euclidean_values)
        }

        summary_rows.append({
            "topic": topic,
            "persona": persona,
            "avg_cosine": avg_cosine,
            "avg_euclidean": avg_euclidean,
            "samples": samples,
            "db_avg_cosine": db_avg_cosine,
            "db_avg_euclidean": db_avg_euclidean,
            "db_samples": db_samples,
        })

    separator = print_separator(False)
    db_header_lines = []
    for db_alias in db_aliases:
        db_cos = db_header_cosine[db_alias]
        db_euc = db_header_euclidean[db_alias]
        db_cos_text = f"{(sum(db_cos) / len(db_cos)):.6f}" if db_cos else "NO DATA"
        db_euc_text = f"{(sum(db_euc) / len(db_euc)):.6f}" if db_euc else "NO DATA"
        db_header_lines.append(
            f"{db_alias} before/after avg: cosine={db_cos_text}, euclidean={db_euc_text}"
        )
    db_persona_lines = []
    for db in db_aliases:
        for persona in persona_order:
            cos_vals = db_persona_cosine[db][persona]
            euc_vals = db_persona_euclidean[db][persona]
            cos_text = f"{(sum(cos_vals) / len(cos_vals)):.6f}" if cos_vals else "NO DATA"
            euc_text = f"{(sum(euc_vals) / len(euc_vals)):.6f}" if euc_vals else "NO DATA"
            db_persona_lines.append(
                f"{db} p{persona} before/after avg: cosine={cos_text}, euclidean={euc_text}"
            )
    header = "\n".join([
        separator,
        f"BATCH SUMMARY (ALL EVENTS) at {get_time()}",
        f"Total events computed: {len(event_outputs)}",
        "Metric basis: persona BEFORE vs AFTER similarities aggregated across events",
        *db_header_lines,
        "DB/PERSONA BEFORE/AFTER AVERAGES:",
        *db_persona_lines,
        f"Embedding Model: {EMBEDDING}",
        separator,
        "AVERAGE SIMILARITY BY TOPIC AND PERSONA",
        "-" * 38,
    ])

    lines = []
    last_topic = None
    for row in summary_rows:
        if last_topic is not None and row["topic"] != last_topic:
            lines.append(separator)
            lines.append("")

        avg_cosine_text = f"{row['avg_cosine']:.6f}" if row["avg_cosine"] is not None else "NO DATA"
        avg_euclidean_text = f"{row['avg_euclidean']:.6f}" if row["avg_euclidean"] is not None else "NO DATA"
        lines.append(
            f"topic = {row['topic']} | persona = p{row['persona']} | "
            f"avg cosine similarity = {avg_cosine_text} | "
            f"avg euclidean distance = {avg_euclidean_text} | "
            f"samples = {row['samples']}"
        )
        last_topic = row["topic"]

    if not lines:
        lines.append("No summary rows could be generated.")

    lines.append(separator)
    summary_text = f"{header}\n" + "\n".join(lines)
    return summary_text, summary_rows

def main_menu() -> int:
    '''Print main menu, validate input, and return choice as int'''

    menu = {
        "1" : "Complete event analysis",
        "2" : "Cross-event analysis",
        "3" : "Custom vector analysis",
        "4" : "Run event analysis for all events (save all)",
        "5" : "Exit"
    }

    # Get and validate input
    while True:
        print("---MAIN MENU---")
        for key, option in menu.items():
            print(f"{key} - {option}")

        choice = input("-> ")
        if choice not in menu:
            clear()
            print(f"\033[33mInvalid\033[0m choice, please enter a number 1 - {len(menu)}\n")
            continue
        break

    return int(choice)

def print_separator(show):
    '''
    Generate a simple display separator
    
    if show is TRUE, display the separator. 
    Otherwise, return it
    '''
    separator = "="*100
    if show:
        print(separator)
    else:
        return separator

def generate_event_report(show, save, event_id, topics, personas, database, before_after, pairwise_before, pairwise_after, pairwise_before_after, pairwise_after_before):
    '''Builds detailed report of given data, which saves and displays as given by save and show respectively'''
    separator = print_separator(False)
    llm_model = "unknown"
    # Try to get LLM model from EVENTS file
    try:
        with open(EVENTS, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("identifier") == event_id:
                    llm_model = row.get("model_name") or "unknown"
                    break
    except Exception:
        pass

    # File header
    header = "\n".join([
        separator,
        f"FULL EVENT COMPUTATION ON {event_id} at {get_time()}",
        f"Personas: {personas}",
        f"Topics: {topics}",
        f"Database: {database}",
        f"LLM Model: {llm_model}",
        f"Embedding Model: {EMBEDDING}",
        separator,
    ])

    # Define computation result types
    result_groups = {
        "before_after": before_after.get(event_id, []),
        "pairwise_before": pairwise_before.get(event_id, []),
        "pairwise_after": pairwise_after.get(event_id, []),
        "pairwise_before_after": pairwise_before_after.get(event_id, []),
        "pairwise_after_before": pairwise_after_before.get(event_id, []),
    }

    # Define result titles
    section_titles = {
        "before_after": "Persona BEFORE vs AFTER",
        "pairwise_before": "Pairwise BEFORE",
        "pairwise_after": "Pairwise AFTER",
        "pairwise_before_after": "Pairwise BEFORE vs AFTER",
        "pairwise_after_before": "Pairwise AFTER vs BEFORE",
    }

    lines = []
    # For each computation title
    for key in section_titles:
        # Get title
        lines.append(section_titles[key])
        lines.append("-" * len(section_titles[key])) # <- Separator, length of title

        rows = result_groups[key]
        if not rows:
            lines.append("No results.\n")
            continue

        last_topic = None
        for row in rows:
            # Add separator between each topic
            if last_topic is not None and row["topic"] != last_topic:
                lines.append(separator)
                lines.append("")

            if "persona" in row:
                who = f"persona={row['persona']}"
            else:
                who = f"persona pair={row['pair']}"

            lines.append(
                f"topic = {row['topic']} | {who} | cosine = {row['cosine']} | euclidean = {row['euclidean']}"
            )
            last_topic = row["topic"]
        lines.append(separator)
        lines.append("")

    results = "\n".join(lines).rstrip()

    # Show output
    if show:
        print(f"{header}\n{results}")
    saved_report_path = None
    if save:
        saved_report_path = save_report(
            f"{header}\n{results}",
            viz_results=result_groups,
            viz_mode="event",
            viz_title=f"Event {event_id}",
        )
    return header, results

def generate_cross_event_report(show, save, event_a_id, event_b_id, topics, personas, database_a, database_b, before_after, pairwise_before, pairwise_after, pairwise_before_after, pairwise_after_before):
    '''Builds detailed report of given data, which saves and displays as given by save and show respectively'''
    separator = print_separator(False)
    llm_model_a = "unknown"
    llm_model_b = "unknown"
    try:
        with open(EVENTS, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                identifier = row.get("identifier")
                if identifier == event_a_id and llm_model_a == "unknown":
                    llm_model_a = row.get("model_name") or "unknown"
                if identifier == event_b_id and llm_model_b == "unknown":
                    llm_model_b = row.get("model_name") or "unknown"
                if llm_model_a != "unknown" and llm_model_b != "unknown":
                    break
    except Exception:
        pass

    # File header
    header = "\n".join([
        separator,
        f"CROSS-EVENT COMPUTATION: {event_a_id} (A) vs {event_b_id} (B) at {get_time()}",
        f"Personas: {personas}",
        f"Topics: {topics}",
        f"Database A: {database_a}",
        f"Database B: {database_b}",
        f"LLM Model A: {llm_model_a}",
        f"LLM Model B: {llm_model_b}",
        f"Embedding Model: {EMBEDDING}",
        separator,
    ])

    result_key = f"{event_a_id}_vs_{event_b_id}"

    result_groups = {
        "before_after": before_after.get(result_key, []),
        "pairwise_before": pairwise_before.get(result_key, []),
        "pairwise_after": pairwise_after.get(result_key, []),
        "pairwise_before_after": pairwise_before_after.get(result_key, []),
        "pairwise_after_before": pairwise_after_before.get(result_key, []),
    }

    section_titles = {
        "before_after": f"Persona BEFORE({event_a_id}) vs AFTER({event_b_id})",
        "pairwise_before": f"Pairwise BEFORE({event_a_id}) vs BEFORE({event_b_id})",
        "pairwise_after": f"Pairwise AFTER({event_a_id}) vs AFTER({event_b_id})",
        "pairwise_before_after": f"Pairwise BEFORE({event_a_id}) vs AFTER({event_b_id})",
        "pairwise_after_before": f"Pairwise AFTER({event_a_id}) vs BEFORE({event_b_id})",
    }

    lines = []
    for key in section_titles:
        lines.append(section_titles[key])
        lines.append("-" * len(section_titles[key]))

        rows = result_groups[key]
        if not rows:
            lines.append("No results.\n")
            continue

        last_topic = None
        for row in rows:
            if last_topic is not None and row["topic"] != last_topic:
                lines.append(separator)
                lines.append("")

            if "persona" in row:
                who = f"persona={row['persona']}"
            else:
                who = f"persona pair={row['pair']}"

            lines.append(
                f"topic = {row['topic']} | {who} | cosine = {row['cosine']} | euclidean = {row['euclidean']}"
            )
            last_topic = row["topic"]
        lines.append(separator)
        lines.append("")

    results = "\n".join(lines).rstrip()

    # Show output
    if show:
        print(f"{header}\n{results}")
    saved_report_path = None
    if save:
        saved_report_path = save_report(
            f"{header}\n{results}",
            viz_results=result_groups,
            viz_mode="cross_event",
            viz_title=f"Cross Event {event_a_id} vs {event_b_id}",
        )

def generate_custom_report(show, save,
                           event_a_id,
                           vector_a_topic,
                           vector_a_persona,
                           vector_a_stage,
                           event_b_id,
                           vector_b_topic,
                           vector_b_persona,
                           vector_b_stage,
                           cosine,
                           euclidean
):
    '''
    Create custom report on computed custom data
    '''
    separator = print_separator(False)
    llm_model_a = "unknown"
    llm_model_b = "unknown"
    # Try to get respective models for events a and b
    try:
        with open(EVENTS, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                identifier = row.get("identifier")
                if identifier == event_a_id and llm_model_a == "unknown":
                    llm_model_a = row.get("model_name") or "unknown"
                if identifier == event_b_id and llm_model_b == "unknown":
                    llm_model_b = row.get("model_name") or "unknown"
                if llm_model_a != "unknown" and llm_model_b != "unknown":
                    break
    except Exception:
        pass

    header = "\n".join([
        separator,
        f"CUSTOM COMPUTATION at {get_time()}",
        f"LLM Model A: {llm_model_a}",
        f"LLM Model B: {llm_model_b}",
        f"Embedding Model: {EMBEDDING}",
        separator,
        f"Vector A: {event_a_id}",
        f"Topic: {vector_a_topic}",
        f"Persona: p{vector_a_persona}",
        f"Stage: {vector_a_stage}",
        f"VECTOR B: {event_b_id}",
        f"Topic: {vector_b_topic}",
        f"Persona: p{vector_b_persona}",
        f"Stage: {vector_b_stage}",
        separator,
    ])

    results = "\n".join([
        f"cosine = {cosine}",
        f"euclidean = {euclidean}",
        separator
    ])

    if show:
        print(f"{header}\n{results}")
    if save:
        saved_report_path = save_report(
            f"{header}\n{results}",
            viz_results={"cosine": cosine, "euclidean": euclidean},
            viz_mode="custom",
            viz_title=f"Custom {event_a_id} vs {event_b_id}",
        )
    return header, results

##### COMPUTATION #####
def compute_cosine_similarity(vector_a, vector_b):
    '''Reshape input vectors as numpy arrays, and compute the cosine similarity of them'''
    a = np.asarray(vector_a).reshape(1, -1)
    b = np.asarray(vector_b).reshape(1, -1)
    similarity = cosine_similarity(a, b)[0][0]
    return float(similarity)

def compute_euclidean_distance(vector_a, vector_b):
    '''Reshape input vectors as numpy arrays and compute the euclidean distance between them'''
    a = np.asarray(vector_a).reshape(1, -1)
    b = np.asarray(vector_b).reshape(1, -1)
    distance = euclidean_distances(a, b)[0][0]
    return float(distance)

def compute_event(event_vectors, show=None, save=None):
    '''Compute the cosine similarities and euclidean distances between vectors of a single event'''
    # Try to get event_id
    try:
        event_id = event_vectors[0]["id"]
    except Exception as e:
        log(base_name, f"{type(e).__name__}: {e}")
        print(f"{type(e).__name__}[ERROR]: {e}")
        return

    log(base_name, f"Running computation on {event_id}")

    # Check VECTORS is not empty
    if not event_vectors:
        message = f"No vectors found for event, skipping analysis"
        log(base_name, message)
        print(message)
        return
    
    # Get event data
    db = event_vectors[0]["db"]
    topics = get_topics(event_vectors)
    personas = get_personas(event_vectors)

    database = get_database(db)

    # Define dicts where computation results will be stored
    before_after_results = {}
    before_after_results[event_id] = []

    pairwise_before_results = {}
    pairwise_before_results[event_id] = []

    pairwise_after_results = {}
    pairwise_after_results[event_id] = []

    pairwise_before_after_results = {}
    pairwise_before_after_results[event_id] = []

    pairwise_after_before_results = {}
    pairwise_after_before_results[event_id] = []
    missing_vector_data = False

    # Compute BEFORE vs AFTER for each persona on each topic
    for topic in topics:
        for persona in personas:
            vector_a = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == persona and row["s"] == "BEFORE"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == persona and row["s"] == "AFTER"),
                None
            )

            # Handle missing vectors (TOS Violation / timeout)
            if vector_a is None or vector_b is None:
                missing_vector_data = True
                before_after_cosine = "NO COMPARISON GENERATED"
                before_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"No comparison generated for event={event_id}, topic={topic}, "
                    f"left=({persona}, BEFORE), right=({persona}, AFTER)"
                )
                log(base_name, message)
            # If both vectors are found, compute
            else:
                before_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                before_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            before_after_results[event_id].append({
                                        "topic": topic,
                                        "persona": persona,
                                        "cosine": before_after_cosine,
                                        "euclidean": before_after_euclidean})
            
    # Compute persona pairwise BEFORE for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            # Get before vector
            vector_a = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "BEFORE"),
                None
            )
            # Get after vector
            vector_b = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "BEFORE"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_before_cosine = "NO COMPARISON GENERATED"
                pair_before_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"No comparison generated for event={event_id}, topic={topic}, "
                    f"left=({p1}, BEFORE), right=({p2}, BEFORE)"
                )
                log(base_name, message)
            else:
                pair_before_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_before_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_before_results[event_id].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_before_cosine,
                                        "euclidean": pair_before_euclidean})
            
    # Compute persona pairwise AFTER for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "AFTER"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "AFTER"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_after_cosine = "NO COMPARISON GENERATED"
                pair_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"No comparison generated for event={event_id}, topic={topic}, "
                    f"left=({p1}, AFTER), right=({p2}, AFTER)"
                )
                log(base_name, message)
            else:
                pair_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_after_results[event_id].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_after_cosine,
                                        "euclidean": pair_after_euclidean})
            
    # Compute persona pairwise BEFORE vs AFTER for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "BEFORE"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "AFTER"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_before_after_cosine = "NO COMPARISON GENERATED"
                pair_before_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"No comparison generated for event={event_id}, topic={topic}, "
                    f"left=({p1}, BEFORE), right=({p2}, AFTER)"
                )
                log(base_name, message)
            else:
                pair_before_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_before_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_before_after_results[event_id].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_before_after_cosine,
                                        "euclidean": pair_before_after_euclidean})
            
    # Compute persona pairwise AFTER vs BEFORE for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "AFTER"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "BEFORE"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_after_before_cosine = "NO COMPARISON GENERATED"
                pair_after_before_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"No comparison generated for event={event_id}, topic={topic}, "
                    f"left=({p1}, AFTER), right=({p2}, BEFORE)"
                )
                log(base_name, message)
            else:
                pair_after_before_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_after_before_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_after_before_results[event_id].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_after_before_cosine,
                                        "euclidean": pair_after_before_euclidean})

    if missing_vector_data:
        print("One or more computations could not be completed due to missing vector data")

    if show is None:
        show = prompt_show()
    if save is None:
        save = propmt_save()
    
    # Generate event report and display/save as chosen
    header, results = generate_event_report(show, save,
                                            event_id,
                                            topics,
                                            personas,
                                            database,
                                            before_after_results,
                                            pairwise_before_results,
                                            pairwise_after_results,
                                            pairwise_before_after_results,
                                            pairwise_after_before_results
                                            )
    return {
        "event_id": event_id,
        "topics": topics,
        "personas": personas,
        "database": database,
        "before_after_results": before_after_results,
        "pairwise_before_results": pairwise_before_results,
        "pairwise_after_results": pairwise_after_results,
        "pairwise_before_after_results": pairwise_before_after_results,
        "pairwise_after_before_results": pairwise_after_before_results,
        "missing_vector_data": missing_vector_data,
        "header": header,
        "results": results,
    }

def compute_all_events(vector_data):
    '''Run event-level computation for every event in VECTORS'''
    log(base_name, "Running event computation for all events")
    event_ids = get_all_event_ids(vector_data)

    # Skip if no events found
    if not event_ids:
        print("No events found in vector data.")
        return

    event_outputs = []
    missing_any_data = False

    for event_id in event_ids:
        event_vectors = select_event_vectors(vector_data, event_id)
        event_output = compute_event(event_vectors, show=False, save=False)
        if event_output is None:
            continue

        event_outputs.append(event_output)
        missing_any_data = missing_any_data or event_output["missing_vector_data"] # <- bool dict key returned from compute_event function

    if not event_outputs:
        print("No event computations were generated.")
        return

    if missing_any_data:
        print("One or more computations could not be completed due to missing vector data")

    print(f"Computed {len(event_outputs)} events.")
    summary_text, summary_rows = build_batch_summary(event_outputs)

    show = prompt_show()
    save = propmt_save()
    output_text = "\n\n".join([f"{row['header']}\n{row['results']}" for row in event_outputs]) + "\n\n" + summary_text

    if show:
        print(output_text)
    if save:
        while True:
            extension = input("Enter output extension (.txt or .pdf) -> ").strip().lower()
            if extension in [".txt", ".pdf"]:
                break
            print("Invalid extension, please enter .txt or .pdf")

        saved_count = 0
        for event_output in event_outputs:
            event_id = event_output["event_id"]
            safe_event_id = clean_filename(event_id)
            filename = f"evt_{safe_event_id}{extension}"
            report_text = f"{event_output['header']}\n{event_output['results']}"
            viz_results = {
                "before_after": event_output["before_after_results"][event_id],
                "pairwise_before": event_output["pairwise_before_results"][event_id],
                "pairwise_after": event_output["pairwise_after_results"][event_id],
                "pairwise_before_after": event_output["pairwise_before_after_results"][event_id],
                "pairwise_after_before": event_output["pairwise_after_before_results"][event_id],
            }

            saved_path = save_report(
                report_text,
                viz_results=viz_results,
                viz_mode="event",
                viz_title=f"Event {event_id}",
                filename=filename,
                clear_after_save=False,
            )
            if saved_path:
                saved_count += 1

        summary_saved = save_report(
            summary_text,
            viz_results={"rows": summary_rows},
            viz_mode="summary",
            viz_title="Batch Summary",
            filename=f"summary{extension}",
            clear_after_save=False,
        )

        if summary_saved:
            print(f"Saved {saved_count} event files and summary file to {REPORT_FOLDER}")
        else:
            print(f"Saved {saved_count} event files to {REPORT_FOLDER}")

def compute_cross_event(event_a_vectors, event_b_vectors):
    '''Compute similarities between event A and event B'''

    try:
        event_a_id = event_a_vectors[0]["id"]
        event_b_id = event_b_vectors[0]["id"]
    except Exception as e:
        log(base_name, str(e))
        print(f"Error parsing {VECTORS}")
        return
    
    log(base_name, f"Running cross computation on {event_a_id}/{event_b_id}")

    # Check vectors are not empty
    if not event_a_vectors or not event_b_vectors:
        message = "No vectors found for one or both events, skipping cross-event analysis"
        log(base_name, message)
        print(message)
        return

    # Get event data (use event A topics/personas to keep same project flow)
    topics = get_topics(event_a_vectors)
    personas = get_personas(event_a_vectors)
    db_a = event_a_vectors[0]["db"]
    db_b = event_b_vectors[0]["db"]
    
    database_a = get_database(db_a)
    database_b = get_database(db_b)

    result_key = f"{event_a_id}_vs_{event_b_id}"

    # Define dicts where computation results will be stored
    before_after_results = {}
    before_after_results[result_key] = []

    pairwise_before_results = {}
    pairwise_before_results[result_key] = []

    pairwise_after_results = {}
    pairwise_after_results[result_key] = []

    pairwise_before_after_results = {}
    pairwise_before_after_results[result_key] = []

    pairwise_after_before_results = {}
    pairwise_after_before_results[result_key] = []
    missing_vector_data = False

    # Compute BEFORE(A) vs AFTER(B) for each persona on each topic
    for topic in topics:
        for persona in personas:
            vector_a = next(
                (row["vector"] for row in event_a_vectors
                 if row["t"] == topic and row["p"] == persona and row["s"] == "BEFORE"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_b_vectors
                 if row["t"] == topic and row["p"] == persona and row["s"] == "AFTER"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                before_after_cosine = "NO COMPARISON GENERATED"
                before_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"Missing vector data: no comparison generated for event_a={event_a_id}, event_b={event_b_id}, topic={topic}, "
                    f"left=({persona}, BEFORE), right=({persona}, AFTER)"
                )
                log(base_name, message)
            else:
                before_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                before_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            before_after_results[result_key].append({
                                        "topic": topic,
                                        "persona": persona,
                                        "cosine": before_after_cosine,
                                        "euclidean": before_after_euclidean})

    # Compute persona pairwise BEFORE(A) vs BEFORE(B) for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_a_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "BEFORE"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_b_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "BEFORE"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_before_cosine = "NO COMPARISON GENERATED"
                pair_before_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"Missing vector data: no comparison generated for event_a={event_a_id}, event_b={event_b_id}, topic={topic}, "
                    f"left=({p1}, BEFORE), right=({p2}, BEFORE)"
                )
                log(base_name, message)
            else:
                pair_before_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_before_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_before_results[result_key].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_before_cosine,
                                        "euclidean": pair_before_euclidean})

    # Compute persona pairwise AFTER(A) vs AFTER(B) for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_a_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "AFTER"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_b_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "AFTER"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_after_cosine = "NO COMPARISON GENERATED"
                pair_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"Missing vector data: no comparison generated for event_a={event_a_id}, event_b={event_b_id}, topic={topic}, "
                    f"left=({p1}, AFTER), right=({p2}, AFTER)"
                )
                log(base_name, message)
            else:
                pair_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_after_results[result_key].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_after_cosine,
                                        "euclidean": pair_after_euclidean})

    # Compute persona pairwise BEFORE(A) vs AFTER(B) for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_a_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "BEFORE"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_b_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "AFTER"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_before_after_cosine = "NO COMPARISON GENERATED"
                pair_before_after_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"Missing vector data: no comparison generated for event_a={event_a_id}, event_b={event_b_id}, topic={topic}, "
                    f"left=({p1}, BEFORE), right=({p2}, AFTER)"
                )
                log(base_name, message)
            else:
                pair_before_after_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_before_after_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_before_after_results[result_key].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_before_after_cosine,
                                        "euclidean": pair_before_after_euclidean})

    # Compute persona pairwise AFTER(A) vs BEFORE(B) for each topic
    for topic in topics:
        for p1, p2 in combinations(personas, 2):
            vector_a = next(
                (row["vector"] for row in event_a_vectors
                 if row["t"] == topic and row["p"] == p1 and row["s"] == "AFTER"),
                None
            )
            vector_b = next(
                (row["vector"] for row in event_b_vectors
                 if row["t"] == topic and row["p"] == p2 and row["s"] == "BEFORE"),
                None
            )

            if vector_a is None or vector_b is None:
                missing_vector_data = True
                pair_after_before_cosine = "NO COMPARISON GENERATED"
                pair_after_before_euclidean = "NO COMPARISON GENERATED"
                message = (
                    f"Missing vector data: no comparison generated for event_a={event_a_id}, event_b={event_b_id}, topic={topic}, "
                    f"left=({p1}, AFTER), right=({p2}, BEFORE)"
                )
                log(base_name, message)
            else:
                pair_after_before_cosine = compute_cosine_similarity(vector_a, vector_b)
                pair_after_before_euclidean = compute_euclidean_distance(vector_a, vector_b)

            pairwise_after_before_results[result_key].append({
                                        "topic": topic,
                                        "pair": (p1, p2),
                                        "cosine": pair_after_before_cosine,
                                        "euclidean": pair_after_before_euclidean})

    if missing_vector_data:
        print("One or more computations could not be completed due to missing vector data")

    show = prompt_show()
    save = propmt_save()
    
    # Generate event report and display/save as chosen
    generate_cross_event_report(show, save,
                        event_a_id,
                        event_b_id,
                        topics,
                        personas,
                        database_a,
                        database_b,
                        before_after_results,
                        pairwise_before_results,
                        pairwise_after_results,
                        pairwise_before_after_results,
                        pairwise_after_before_results
                        )

def compute_custom(vector_data):
    ''''Get 2 custom vectors and compute cosine similarity and euclidean distance'''
    log(base_name, "Running custom computation")
    vector_a_found = False
    vector_b_found = False
    
    # Data stages (intervention stage)
    stages = ['BEFORE', 'AFTER']

    # Try to find vector A
    while not vector_a_found:
        # Get event data
        event_a_id = get_event_selection(vector_data, cross_choice="A", cross_compute=True)
        event_a_vectors = select_event_vectors(vector_data, event_a_id)
        event_a_topics = get_topics(event_a_vectors)
        event_a_personas = get_personas(event_a_vectors)

        # Get topic
        while True:
            print("Select topic for A")
            for choice, topic in enumerate(event_a_topics, start=1):
                print(f"{choice} - {topic}")
            choice = input("-> ")

            # Check numeric
            try: 
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(event_a_topics)}")
                continue

            # Check in range
            if choice not in range(1, len(event_a_topics) + 1):
                print(f"Invalid, select a number 1-{len(event_a_topics)}")
                continue
            vector_a_topic = event_a_topics[choice-1]
            break
        
        # Get persona
        while True:
            print("Select persona for A")
            for choice, persona in enumerate(event_a_personas, start=1):
                print(f"{choice} - p{persona}")
            choice = input("-> ")

            # Check numeric
            try:
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(event_a_personas)}")
                continue

            # Check in range
            if choice not in range(1, len(event_a_personas) + 1):
                print(f"Invalid, you must select a number 1-{len(event_a_personas)}")
                continue
            vector_a_persona = event_a_personas[choice-1]
            break

        # Get state
        while True:
            print("Select stage for A")
            for choice, stage in enumerate(stages, start=1):
                print(f"{choice} - {stage}")
            choice = input("-> ")

            # Check numeric
            try:
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(stages)}")
                continue

            # Check in range
            if choice not in range(1, len(stages) + 1):
                print(f"Invalid, you must select a number 1-{len(stages)}")
                continue
            vector_a_stage = stages[choice-1]
            break

        vector_a = get_vector(event_a_vectors, vector_a_topic, vector_a_persona, vector_a_stage)

        if vector_a:
            vector_a_found = True
        else:
            print("Vector A not found for selected values, please try again.")

    # Try to find vector B        
    while not vector_b_found:
        event_b_id = get_event_selection(vector_data, cross_choice="B", cross_compute=True)
        event_b_vectors = select_event_vectors(vector_data, event_b_id)
        event_b_topics = get_topics(event_b_vectors)
        event_b_personas = get_personas(event_b_vectors)

        while True:
            print("Select topic for B")
            for choice, topic in enumerate(event_b_topics, start=1):
                print(f"{choice} - {topic}")
            choice = input("-> ")

            try:
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(event_b_topics)}")
                continue

            if choice not in range(1, len(event_b_topics) + 1):
                print(f"Invalid, select a number 1-{len(event_b_topics)}")
                continue
            vector_b_topic = event_b_topics[choice-1]
            break

        while True:
            print("Select persona for B")
            for choice, persona in enumerate(event_b_personas, start=1):
                print(f"{choice} - p{persona}")
            choice = input("-> ")

            try:
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(event_b_personas)}")
                continue

            if choice not in range(1, len(event_b_personas) + 1):
                print(f"Invalid, you must select a number 1-{len(event_b_personas)}")
                continue
            vector_b_persona = event_b_personas[choice-1]
            break

        while True:
            print("Select stage for B")
            for choice, stage in enumerate(stages, start=1):
                print(f"{choice} - {stage}")
            choice = input("-> ")

            try:
                choice = int(choice)
            except ValueError:
                print(f"Invalid, you must select a number 1-{len(stages)}")
                continue

            if choice not in range(1, len(stages) + 1):
                print(f"Invalid, you must select a number 1-{len(stages)}")
                continue
            vector_b_stage = stages[choice-1]
            break

        vector_b = get_vector(event_b_vectors, vector_b_topic, vector_b_persona, vector_b_stage)

        if vector_b:
            vector_b_found = True
        else:
            print("Vector B not found for selected values, please try again.")

    # Compute values for vectors A and B
    cosine = compute_cosine_similarity(vector_a, vector_b)
    euclidean = compute_euclidean_distance(vector_a, vector_b)
    
    show = prompt_show()
    save = propmt_save()

    generate_custom_report(show, save,
                           event_a_id,
                           vector_a_topic,
                           vector_a_persona,
                           vector_a_stage,
                           event_b_id,
                           vector_b_topic,
                           vector_b_persona,
                           vector_b_stage,
                           cosine,
                           euclidean
                           ) 

#### MAIN ####
def main():

    while True:
        vector_data = save_rows_to_dict()
        match main_menu():

            # Complete event analysis
            case 1:
                clear()
                print("--Complete Event Analysis--")
                event_id = get_event_selection(vector_data)
                event_vectors = select_event_vectors(vector_data, event_id)
                compute_event(event_vectors)
                print()
            # Cross event analysis
            case 2:
                clear()
                print("--Cross Event Analysis--")
                event_a_id = get_event_selection(vector_data, cross_compute=True, cross_choice="A")
                event_b_id = get_event_selection(vector_data, cross_compute=True, cross_choice="B")
                event_a_vectors = select_event_vectors(vector_data, event_a_id)
                event_b_vectors = select_event_vectors(vector_data, event_b_id)
                compute_cross_event(event_a_vectors, event_b_vectors)
                print()
            # Custom analysis
            case 3:
                clear()
                print("--Custom Analysis--")
                compute_custom(vector_data)
                print()
            # All event analysis
            case 4:
                clear()
                print("--All Event Analysis--")
                compute_all_events(vector_data)
                print()
            # Exit
            case 5:
                print("Exiting")
                return

##### Execute main #####
if __name__ == "__main__":
    try:
        log(base_name, "Session started")
        main()
        log(base_name, "Session completed")
    except KeyboardInterrupt as k:
        log(base_name, f"{type(k).__name__}: Session terminated")
        print("Terminating session")
    except Exception as e:
        log(base_name, f"{type(e).__name__}: {e}")
        print("\033[33mAn error occured, check log.jsonl for more information. \033[31mTerminating session.\033[0m")
