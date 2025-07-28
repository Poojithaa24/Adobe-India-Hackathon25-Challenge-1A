import fitz  # PyMuPDF
import re

def clean_repeated_chunks(text):
    # Remove duplicate words like "Proposal Proposal" → "Proposal"
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def vertical_center(line):
    return line["y0"] + line["font_size"] / 2

def extract_pdf_layout(pdf_path):
    doc = fitz.open(pdf_path)
    lines_by_page = {}

    for page_num, page in enumerate(doc):
        page_lines = []
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = " ".join([span.get("text", "") for span in line.get("spans", [])]).strip()
                    if not line_text:
                        continue

                    first_span = line.get("spans", [{}])[0]
                    font_size = round(first_span.get("size", 0), 2)
                    font_name = first_span.get("font", "")
                    is_bold = "bold" in font_name.lower()
                    x0, y0, _, _ = first_span.get("bbox", (0, 0, 0, 0))

                    page_lines.append({
                        "text": line_text,
                        "font_size": font_size,
                        "font_name": font_name,
                        "is_bold": is_bold,
                        "x0": x0,
                        "y0": y0,
                        "page_number": page_num + 1,
                    })
        lines_by_page[page_num + 1] = page_lines

    # --- Aggressive Line Consolidation ---
    consolidated_lines = []
    page_text_map = {}

    for page_num, lines in lines_by_page.items():
        if not lines:
            continue

        current_line_idx = 0
        while current_line_idx < len(lines):
            base_line = lines[current_line_idx]
            next_line_idx = current_line_idx + 1

            while next_line_idx < len(lines):
                next_line = lines[next_line_idx]
                vertical_distance = abs(vertical_center(next_line) - vertical_center(base_line))

                if (
                    next_line["font_name"] == base_line["font_name"]
                    and abs(next_line["font_size"] - base_line["font_size"]) < 1
                    and vertical_distance < max(base_line["font_size"], next_line["font_size"]) * 1.2
                    and next_line["text"] not in base_line["text"]
                ):
                    base_line["text"] += " " + next_line["text"]
                    next_line_idx += 1
                else:
                    break

            final_text = clean_repeated_chunks(" ".join(base_line["text"].strip().split()))
            base_line["text"] = final_text
            base_line["text_length"] = len(final_text.split())
            base_line["is_uppercase"] = final_text.isupper()
            base_line["line_indent"] = base_line["x0"]
            base_line["ends_with_period"] = final_text.endswith(".")

            page_text_map[final_text] = page_text_map.get(final_text, 0) + 1
            consolidated_lines.append(base_line)

            current_line_idx = next_line_idx

    # Add frequency info
    for line in consolidated_lines:
        line["appears_on_many_pages"] = page_text_map.get(line["text"], 0) > 2

    return consolidated_lines
