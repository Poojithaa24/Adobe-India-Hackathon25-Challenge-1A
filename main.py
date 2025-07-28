import os
import json
import joblib
from collections import Counter
import pandas as pd

# Make sure to have these functions available in your project
from extractor import extract_pdf_layout
from nlp_utils import compute_heading_score

INPUT_DIR = "input"
MODEL_PATH = "classifier_1.pkl"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classifier = joblib.load(MODEL_PATH)

feature_names = [
    "font_size", "is_bold", "text_length", "x0", "y0",
    "is_uppercase", "line_indent", "ends_with_period", "page_number",
    "appears_on_many_pages", "heading_score"
]

def extract_features(line):
    """Extracts features for the classifier from a line of text."""
    try:
        return [
            line["font_size"], int(line["is_bold"]), line["text_length"],
            line["x0"], line["y0"], int(line["is_uppercase"]),
            line["line_indent"], int(line["ends_with_period"]),
            line["page_number"], int(line["appears_on_many_pages"]),
            compute_heading_score(line["text"]),
        ]
    except KeyError as e:
        # Silently skip lines with missing keys
        return None

def consolidate_headings(headings, y_threshold=10):
    """Merges multi-line headings into a single entry."""
    if not headings:
        return []

    consolidated = []
    current_heading = headings[0]

    for next_heading in headings[1:]:
        same_level = next_heading["level"] == current_heading["level"]
        same_page = next_heading["page"] == current_heading["page"]
        close_vertically = abs(next_heading["y"] - current_heading["y"]) < y_threshold

        if same_level and same_page and close_vertically:
            current_heading["text"] += " " + next_heading["text"]
        else:
            consolidated.append(current_heading)
            current_heading = next_heading

    consolidated.append(current_heading)
    return consolidated

def map_font_sizes_to_levels(lines):
    """
    IMPROVED HEURISTIC: Creates a mapping from text style (font size, bold) to a heading level.
    """
    style_counter = Counter(
        (line["font_size"], line.get("is_bold", False)) for line in lines
    )

    if not style_counter:
        return {}

    body_style = style_counter.most_common(1)[0][0]
    body_font_size = body_style[0]

    heading_styles = [
        style for style, count in style_counter.items()
        if style[0] > body_font_size or (style[0] == body_font_size and style[1])
    ]

    heading_styles.sort(key=lambda x: (x[0], x[1]), reverse=True)

    style_to_level = {}
    level_map = ["H1", "H2", "H3"]
    for i, style in enumerate(heading_styles):
        if i < len(level_map):
            style_to_level[style] = level_map[i]
        else:
            break

    return style_to_level

def process_pdf(pdf_path):
    """Processes a single PDF to extract its title and outline."""
    lines = extract_pdf_layout(pdf_path)
    if not lines:
        return

    style_level_mapping = map_font_sizes_to_levels(lines)
    all_predicted_headings = []

    for line in lines:
        text = line["text"].strip()
        if len(text) < 3 or not any(c.isalpha() for c in text):
            continue

        features = extract_features(line)
        if features is None:
            continue

        # Convert to a DataFrame for compatibility with the model's feature names expectation
        X = pd.DataFrame([features], columns=feature_names)
        model_probs = classifier.predict_proba(X)[0]
        model_label = classifier.predict(X)[0]
        model_conf = max(model_probs)

        style_key = (line["font_size"], line.get("is_bold", False))
        heuristic_label = style_level_mapping.get(style_key, "body")

        final_label = "body"
        if model_label == "title":
            final_label = "title"
        elif model_conf > 0.9:
            final_label = model_label
        elif heuristic_label in ["H1", "H2", "H3"]:
            final_label = heuristic_label
        elif model_conf > 0.6:
            final_label = model_label

        if final_label not in ["not_heading", "content", "footer", "body"]:
            all_predicted_headings.append({
                "level": final_label, "text": text, "page": line["page_number"],
                "y": line.get("y0", 0)
            })

    title_candidates = [h for h in all_predicted_headings if h["level"] == "title"]
    if title_candidates:
        title_candidates.sort(key=lambda h: (h["page"], -h.get("y", 0)))
        final_title = " ".join([h["text"].strip() for h in title_candidates])
    else:
        final_title = ""

    outline_headings = [h for h in all_predicted_headings if h["level"] != "title"]
    outline_headings.sort(key=lambda h: (h["page"], h["y"]))
    clean_outline = consolidate_headings(outline_headings)

    result = {
        "title": final_title.strip(),
        "outline": [
            {
                "level": h["level"],
                "text": h["text"].strip(),
                "page": h["page"]
            }
            for h in clean_outline if any(c.isalnum() for c in h["text"])
        ]
    }

    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def main():
    """Main function to find and process all PDFs in the input directory."""
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return

    for pdf_file in pdf_files:
        full_path = os.path.join(INPUT_DIR, pdf_file)
        try:
            process_pdf(full_path)
        except Exception as e:
            # Still print critical errors for troubleshooting
            print(f"❌ An unexpected error occurred while processing {pdf_file}: {e}")

if __name__ == "__main__":
    main()