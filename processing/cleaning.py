import re
import csv

# Load raw output from the scraper
with open("essays_output.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

paragraphs = []
current_paragraph = []

for line in lines:
    line = line.strip()

    # Skip metadata, links, and junk lines
    if (line.startswith("---") or
        line.startswith("* [http") or
        line.startswith("[") or
        line.startswith("©") or
        line == "|" or
        line == "" or
        re.match(r"^\[\d+\]$", line) or
            re.match(r"^\[\s*\]$", line)):
        if current_paragraph:
            joined = " ".join(current_paragraph).strip()
            if len(joined.split()) > 10:
                paragraphs.append(joined)
            current_paragraph = []
        continue

    current_paragraph.append(line)

# Add the final paragraph
if current_paragraph:
    joined = " ".join(current_paragraph).strip()
    if len(joined.split()) > 10:
        paragraphs.append(joined)

# Deduplicate
unique_paragraphs = list(dict.fromkeys(paragraphs))

# Write to CSV
with open("pg_paragraphs.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["paragraph"])
    for p in unique_paragraphs:
        writer.writerow([p])

print(
    f"✨ Done. Extracted {len(unique_paragraphs)} unique paragraphs to 'pg_paragraphs.csv'")
