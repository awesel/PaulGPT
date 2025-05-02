import requests
from bs4 import BeautifulSoup
import re
import time
import os  # To help suggest an output filename

# --- Function to extract essay text ---
# Modified to accept an optional 'outfile' argument for writing status messages


def extract_essay_text(url, outfile=None):
    """
    Fetches a URL, parses HTML, and attempts to extract the main text content.
    Writes status messages to 'outfile' if provided.
    Returns the extracted text or an error message string.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    status_prefix = "   "  # Indentation for status messages in the file

    def write_status(message):
        """Helper to write status messages to file if available, otherwise print."""
        full_message = status_prefix + message + "\n"
        if outfile:
            try:
                outfile.write(full_message)
            except Exception as e:
                print(f"Console fallback: Error writing status to file: {e}")
                # Print to console if file write fails
                print(full_message, end='')
        else:
            # Print to console if no outfile provided
            print(full_message, end='')

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        main_content = None
        selectors_to_try = [
            'article', 'main', 'div[role="main"]', '#content', '#main-content',
            '#article-body', '.entry-content', '.post-content', '.article-content',
            '.main-content', '.td-post-content',
        ]

        for selector in selectors_to_try:
            # Simplified selector logic (bs4 handles basic CSS selectors)
            try:
                # Use select_one which handles tags, IDs, classes, and attributes
                main_content = soup.select_one(selector)
                if main_content:
                    write_status(
                        f"[Info] Found content using selector: '{selector}'")
                    break
            except NotImplementedError:
                write_status(
                    f"[Warning] Selector '{selector}' not supported by select_one, skipping.")
                continue  # Skip unsupported selectors

        if main_content:
            for element in main_content(["script", "style", "nav", "footer", "aside", "form"]):
                element.decompose()
            text = main_content.get_text(separator='\n\n', strip=True)
        else:
            write_status(
                "[Warning] Could not find specific main content tag. Falling back to all <p> tags in <body>.")
            body_paragraphs = soup.body.find_all('p') if soup.body else []
            if body_paragraphs:
                text = "\n\n".join(p.get_text(strip=True)
                                   for p in body_paragraphs if p.get_text(strip=True))
                if not text:  # Handle case where <p> tags were empty
                    write_status(
                        "[Warning] Found <p> tags but they contained no text. Falling back to body text.")
                    text = soup.body.get_text(
                        separator='\n\n', strip=True) if soup.body else "--- Could not extract any text ---"
            else:
                write_status(
                    "[Warning] No <p> tags found or body tag missing. Falling back to all body text (might be noisy).")
                text = soup.body.get_text(
                    separator='\n\n', strip=True) if soup.body else "--- Could not extract any text ---"

        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up extra blank lines
        return text if text else "--- No meaningful text extracted ---"

    except requests.exceptions.Timeout:
        error_msg = f"--- Error: Request timed out for {url} ---"
        write_status(error_msg)  # Write error status to file too
        return error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"--- Error fetching {url}: {e} ---"
        write_status(error_msg)
        return error_msg
    except Exception as e:
        # Catch potential errors during parsing or text extraction as well
        error_msg = f"--- Error processing {url}: {e} ---"
        write_status(error_msg)
        return error_msg

# --- Function to process the links file ---
# Modified to accept output_filepath and write results there


def process_links_file(input_filepath, output_filepath):
    """
    Reads an input file, extracts URLs, fetches essays, and writes all
    output (status, essays, errors) to the output file.
    Prints progress updates to the console.
    """
    link_pattern = re.compile(r'\* \[([^\]]+)\]')
    lines_processed = 0
    links_found = 0

    print(
        f"Starting processing. Input: '{input_filepath}', Output: '{output_filepath}'")

    try:
        # Open both files using 'with' for automatic closing
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w', encoding='utf-8') as outfile:

            outfile.write(f"--- Processing Log and Extracted Text ---\n")
            outfile.write(f"Input File: {input_filepath}\n")
            outfile.write(f"Output File: {output_filepath}\n")
            outfile.write(
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            outfile.write("="*60 + "\n")

            for i, line in enumerate(infile):
                lines_processed += 1
                line = line.strip()
                match = link_pattern.search(line)

                if match:
                    links_found += 1
                    url = match.group(1)
                    # Print progress to console
                    print(f"Processing Line {i+1}: Found URL: {url}")
                    # Write detailed info to file
                    outfile.write(
                        f"\n--- Processing Line {i+1}: Found URL: {url} ---\n")
                    outfile.write(f"Original Line: {line}\n")

                    # Pass the outfile object to the extraction function
                    essay_or_error = extract_essay_text(url, outfile)

                    outfile.write("\n--- Essay Text / Result ---\n")
                    # Add newline after essay/error
                    outfile.write(essay_or_error + "\n")
                    outfile.write("--- End of Entry ---\n")
                    outfile.write("="*60 + "\n")  # Separator in file

                    # Polite delay
                    time.sleep(1)

                elif line:  # Non-empty line that didn't match
                    # Optional: Log skipped lines to the output file
                    outfile.write(
                        f"\n--- Skipping Line {i+1}: No matching URL pattern found ---\n")
                    outfile.write(f"   Line content: {line}\n")
                    outfile.write("="*60 + "\n")

            outfile.write("\n--- Processing Complete ---\n")
            outfile.write(f"Total lines processed: {lines_processed}\n")
            outfile.write(f"Links found and attempted: {links_found}\n")

        print(f"\nProcessing finished. Output saved to '{output_filepath}'")
        print(
            f"Total lines processed: {lines_processed}, Links attempted: {links_found}")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filepath}' was not found.")
    except IOError as e:
        print(
            f"Error: Could not write to output file '{output_filepath}'. Check permissions or path. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    input_file = input(
        "Enter the path to your input text file (e.g., links.txt): ")

    # Suggest an output filename based on the input filename
    default_output_file = os.path.splitext(input_file)[0] + "_output.txt"
    output_file = input(
        f"Enter the path for the output file (press Enter for '{default_output_file}'): ")
    if not output_file:  # If user just presses Enter
        output_file = default_output_file

    process_links_file(input_file, output_file)
