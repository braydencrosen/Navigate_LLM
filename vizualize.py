import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

##### COLORS #####
COLOR_COS = "#2E86AB"
COLOR_EUC = "#E07A2D"
COLOR_POINTS = "#1F2A44"
PERSONA_COLORS = ["#2E86AB", "#E07A2D", "#5E9C36", "#7A5EA6", "#3B9C9C", "#B35C5C"]

##### HELPERS #####
def to_float(value):
    '''Try to parse a float, return None for invalid values'''
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None

def clean_filename(value):
    '''Convert text to a filesystem-safe lowercase filename'''
    text = str(value).strip()
    cleaned = re.sub(r"[^\w\s-]", "", text)
    cleaned = re.sub(r"[\s_]+", "-", cleaned).strip("-").lower()
    return cleaned or "computation"

def save_figure(fig, output_dir, filename):
    '''Save a matplotlib figure and return the full output path'''
    output_path = Path(output_dir) / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

def persona_key(value):
    '''Sort personas numerically when possible, else as text.'''
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))

def point_offsets(count):
    '''Small horizontal spread so overlapping scatter points remain visible.'''
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    step = 0.16 / (count - 1)
    return [-0.08 + (step * idx) for idx in range(count)]

##### BUILDERS #####
def build_persona_before_after_boxplots(results, title_prefix, output_dir, stem):
    '''Build combined cosine/euclidean boxplots by persona. Return the figure if it generated successfully'''
    if not isinstance(results, dict):
        return []

    rows = results.get("before_after", [])
    if not isinstance(rows, list) or not rows:
        return []

    personas = sorted(
        {row.get("persona") for row in rows if row.get("persona") is not None},
        key=persona_key,
    )

    # Build series for each persona
    labels = []
    cosine_series = []
    euclidean_series = []

    for persona in personas:
        cos_vals = []
        euc_vals = []

        for row in rows:
            if row.get("persona") != persona:
                continue

            cos = to_float(row.get("cosine"))
            euc = to_float(row.get("euclidean"))

            if cos is not None:
                cos_vals.append(cos)
            if euc is not None:
                euc_vals.append(euc)

        # Skip personas that have no values
        if not cos_vals and not euc_vals:
            continue

        labels.append(f"P{persona}")
        cosine_series.append(cos_vals if cos_vals else [0.0])
        euclidean_series.append(euc_vals if euc_vals else [0.0])

    if not labels:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # Cosine plot
    bp_cos = axes[0].boxplot(cosine_series, patch_artist=True, showmeans=True, labels=labels)
    for box in bp_cos["boxes"]:
        box.set_facecolor(COLOR_COS)
        box.set_alpha(0.45)
    for median in bp_cos["medians"]:
        median.set_color("#1C4E80")
        median.set_linewidth(1.6)

    for i, values in enumerate(cosine_series, start=1):
        if not values:
            continue
        offsets = point_offsets(len(values))
        axes[0].scatter([i + off for off in offsets], values, s=18, alpha=0.65, color=COLOR_POINTS, zorder=3)

    axes[0].set_title(f"{title_prefix} - Cosine Similarity (BEFORE vs AFTER)")
    axes[0].set_xlabel("Persona")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Euclidean plot
    bp_euc = axes[1].boxplot(euclidean_series, patch_artist=True, showmeans=True, labels=labels)
    for box in bp_euc["boxes"]:
        box.set_facecolor(COLOR_EUC)
        box.set_alpha(0.45)
    for median in bp_euc["medians"]:
        median.set_color("#8B3A0E")
        median.set_linewidth(1.6)

    for i, values in enumerate(euclidean_series, start=1):
        if not values:
            continue
        offsets = point_offsets(len(values))
        axes[1].scatter([i + off for off in offsets], values, s=18, alpha=0.65, color=COLOR_POINTS, zorder=3)

    axes[1].set_title(f"{title_prefix} - Euclidean Distance (BEFORE vs AFTER)")
    axes[1].set_xlabel("Persona")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"{title_prefix} - Persona Distributions", y=0.99, fontsize=13)
    return [save_figure(fig, output_dir, f"{stem}_persona_before_after_boxplots.png")]

def build_topic_persona_bars(results, title_prefix, output_dir, stem):
    '''Build grouped bars for topic/persona averages (cosine + euclidean). Return plot as a figure if it generates successfully.'''
    if not isinstance(results, dict):
        return []

    rows = results.get("before_after", [])
    if not isinstance(rows, list) or not rows:
        return []

    topics = []
    seen_topics = set()
    for row in rows:
        topic = row.get("topic")
        if topic is not None and topic not in seen_topics:
            seen_topics.add(topic)
            topics.append(topic)

    personas = sorted(
        {row.get("persona") for row in rows if row.get("persona") is not None},
        key=persona_key,
    )

    if not topics or not personas:
        return []

    # Store all values per (topic, persona), then average them
    cos_map = {(topic, persona): [] for topic in topics for persona in personas}
    euc_map = {(topic, persona): [] for topic in topics for persona in personas}

    for row in rows:
        key = (row.get("topic"), row.get("persona"))
        if key not in cos_map:
            continue

        cos = to_float(row.get("cosine"))
        euc = to_float(row.get("euclidean"))

        if cos is not None:
            cos_map[key].append(cos)
        if euc is not None:
            euc_map[key].append(euc)

    persona_color_map = {
        persona: PERSONA_COLORS[idx % len(PERSONA_COLORS)]
        for idx, persona in enumerate(personas)
    }

    # Flatten grouped bars into a single x-axis with gaps between topics
    x_positions = []
    x_labels = []
    cos_values = []
    euc_values = []
    bar_colors = []
    topic_centers = []

    x = 0.0
    gap = 1.0

    for topic in topics:
        group_start = x

        for persona in personas:
            key = (topic, persona)
            cos_vals = cos_map[key]
            euc_vals = euc_map[key]

            x_positions.append(x)
            x_labels.append(f"P{persona}")
            cos_values.append(sum(cos_vals) / len(cos_vals) if cos_vals else 0.0)
            euc_values.append(sum(euc_vals) / len(euc_vals) if euc_vals else 0.0)
            bar_colors.append(persona_color_map[persona])
            x += 1.0

        group_end = x - 1.0
        topic_centers.append((group_start + group_end) / 2.0)
        x += gap

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].bar(x_positions, cos_values, width=0.8, color=bar_colors, edgecolor="#333333", linewidth=0.4)
    axes[0].set_title(f"{title_prefix} - BEFORE vs AFTER by Topic/Persona (Cosine)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_xlabel("Persona Within Topic")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(x_labels, rotation=65, ha="right", fontsize=8)
    axes[0].tick_params(labelbottom=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x_positions, euc_values, width=0.8, color=bar_colors, edgecolor="#333333", linewidth=0.4)
    axes[1].set_title(f"{title_prefix} - BEFORE vs AFTER by Topic/Persona (Euclidean)")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].set_xlabel("Persona Within Topic")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels, rotation=65, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    # Write topic names above each group
    for ax in axes:
        y0, y1 = ax.get_ylim()
        y_text = y1 - (y1 - y0) * 0.05
        for center, topic in zip(topic_centers, topics):
            ax.text(center, y_text, str(topic), ha="center", va="top", fontsize=9)

    return [save_figure(fig, output_dir, f"{stem}_topic_persona_before_after_bars.png")]

def build_persona_avg_cosine_bars(results, title_prefix, output_dir, stem):
    '''Build average cosine bars by persona with a best-fit line, return the plot as a figure if it generates successfully'''
    if not isinstance(results, dict):
        return []

    rows = results.get("before_after", [])
    if not isinstance(rows, list) or not rows:
        return []

    values_by_persona = {}
    for row in rows:
        persona = row.get("persona")
        cos = to_float(row.get("cosine"))
        if persona is None or cos is None:
            continue
        values_by_persona.setdefault(persona, []).append(cos)

    if not values_by_persona:
        return []

    personas = sorted(values_by_persona.keys(), key=persona_key)
    x = list(range(1, len(personas) + 1))
    labels = [f"P{p}" for p in personas]
    means = [sum(values_by_persona[p]) / len(values_by_persona[p]) for p in personas]

    # Build least-squares line
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(means)
    sum_xy = sum(px * py for px, py in zip(x, means))
    sum_xx = sum(px * px for px in x)
    denom = (n * sum_xx) - (sum_x * sum_x)

    if denom == 0:
        fit = means[:]
    else:
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
        intercept = (sum_y - (slope * sum_x)) / n
        fit = [intercept + slope * px for px in x]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, means, width=0.7, color=COLOR_COS, alpha=0.75, edgecolor="#1C4E80", linewidth=0.8)
    ax.plot(x, fit, color="#C62828", linewidth=2.2, marker="o", markersize=4, label="Best-Fit Line")
    ax.set_title(f"{title_prefix} - Average Cosine Similarity by Persona")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    # Label each bar top with its mean average
    for x_val, y_val in zip(x, means):
        ax.text(x_val, y_val, f"{y_val:.3f}", ha="center", va="bottom", fontsize=8)

    return [save_figure(fig, output_dir, f"{stem}_persona_avg_cosine_bars.png")]

def build_custom_plot(results, title_prefix, output_dir, stem):
    '''Build a simple custom comparison plot (cosine + euclidean). Return the plot as a figure if it generates successfully'''
    if not isinstance(results, dict):
        return []

    cos = to_float(results.get("cosine"))
    euc = to_float(results.get("euclidean"))
    if cos is None and euc is None:
        return []

    labels = []
    values = []

    if cos is not None:
        labels.append("Cosine Similarity")
        values.append(cos)
    if euc is not None:
        labels.append("Euclidean Distance")
        values.append(euc)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_title(f"{title_prefix} - Custom Comparison")
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        y = bar.get_height()
        ax.annotate(
            f"{y:.4f}",
            (bar.get_x() + bar.get_width() / 2, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    return [save_figure(fig, output_dir, f"{stem}_custom_metrics.png")]

##### EXTERNAL CALL #####
def generate_visualizations(results, saved_report_path, mode="event", title_prefix="Computation"):
    '''Generate visualization files and return their saved paths'''
    if not results or not saved_report_path:
        return []

    report_path = Path(saved_report_path)
    output_dir = report_path.parent / f"{report_path.stem}_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = clean_filename(report_path.stem)

    if mode == "custom":
        return build_custom_plot(results, title_prefix, output_dir, stem)

    if mode == "all_events":
        if not isinstance(results, dict):
            return []

        event_entries = results.get("events", [])
        if not isinstance(event_entries, list):
            return []

        output_paths = []
        for idx, entry in enumerate(event_entries, start=1):
            if not isinstance(entry, dict):
                continue

            event_id = entry.get("event_id", f"event_{idx}")
            event_results = entry.get("results", {})
            event_stem = clean_filename(f"{stem}_{idx}_{event_id}")
            event_title = f"{title_prefix} - Event {event_id}"

            output_paths.extend(build_persona_before_after_boxplots(event_results, event_title, output_dir, event_stem))
            output_paths.extend(build_topic_persona_bars(event_results, event_title, output_dir, event_stem))
            output_paths.extend(build_persona_avg_cosine_bars(event_results, event_title, output_dir, event_stem))
        return output_paths

    if mode == "summary":
        if not isinstance(results, dict):
            return []

        rows = results.get("rows", [])
        if not isinstance(rows, list):
            return []

        summary_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            summary_rows.append(
                {
                    "topic": row.get("topic"),
                    "persona": row.get("persona"),
                    "cosine": row.get("avg_cosine"),
                    "euclidean": row.get("avg_euclidean"),
                }
            )

        payload = {"before_after": summary_rows}
        output_paths = []
        output_paths.extend(build_topic_persona_bars(payload, title_prefix, output_dir, stem))
        output_paths.extend(build_persona_avg_cosine_bars(payload, title_prefix, output_dir, stem))
        return output_paths

    # Default "event" mode
    output_paths = []
    output_paths.extend(build_persona_before_after_boxplots(results, title_prefix, output_dir, stem))
    output_paths.extend(build_topic_persona_bars(results, title_prefix, output_dir, stem))
    output_paths.extend(build_persona_avg_cosine_bars(results, title_prefix, output_dir, stem))
    return output_paths
