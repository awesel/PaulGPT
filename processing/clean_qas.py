import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import re  # For parsing the decision tag more robustly

# --- Configuration ---
INPUT_CSV_FILE = "pg_question_pairs.csv"
OUTPUT_CSV_FILE = "pg_question_pairs_processed.csv"
OPENAI_MODEL = "gpt-4o-mini"  # Specify the model
MAX_RETRIES = 3  # Number of retries for API calls
RETRY_DELAY = 5  # Seconds to wait between retries
# ---------------------


def load_api_key():
    """Loads the OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment variables. "
                         "Please ensure it is set.")
    return api_key


def build_prompt(paragraph, question):
    """Builds the prompt for the OpenAI API call."""
    return f"""Given the following paragraph and question:

Question: "{question}"
Paragraph: "{paragraph}"

Please perform the following tasks:
1. Review the paragraph. Check if it starts correctly (not mid-sentence), is well-formatted, and clearly answers the question.
2. If the paragraph has formatting or grammar issues, make minor edits to make the entry grammatically correct. Do not change the voice of the author or make other edits besides grammar.
3. If the paragraph is already well-formatted and a good answer, you can return it as is. Do not edit anything besides grammar.
4. After providing the final (potentially corrected/rewritten) paragraph, append a decision tag on a NEW LINE indicating whether THIS final paragraph is a good response to the question. Use the exact format <decision>YES</decision> or <decision>NO</decision>.

Provide only the final paragraph followed immediately by the decision tag on the next line. Do not add any other explanations before or after.

Example Output Structure:
[Corrected/Final Paragraph Text Here]
<decision>YES</decision>
"""


def call_openai_api(client, paragraph, question):
    """Calls the OpenAI API with retries and returns the processed content."""
    prompt = build_prompt(paragraph, question)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert editor. Correct the provided paragraph to be a well-formatted and clear answer to the question. Then evaluate if the final paragraph is a good response."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more deterministic corrections
                max_tokens=600  # Adjust based on expected paragraph length + tag
            )
            content = response.choices[0].message.content.strip()
            return content  # Success
        except Exception as e:
            retries += 1
            print(
                f"Error calling OpenAI API (Attempt {retries}/{MAX_RETRIES}): {e}")
            if retries >= MAX_RETRIES:
                print("Max retries reached. Skipping this row.")
                return None  # Indicate failure
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    return None  # Should not be reached if MAX_RETRIES > 0


def parse_openai_response(response_content):
    """Parses the OpenAI response to extract the paragraph and decision."""
    if not response_content:
        return "API_ERROR", "ERROR"

    # Try splitting by newline first, assuming tag is on the last line
    lines = response_content.strip().split('\n')
    last_line = lines[-1].strip()

    decision = "PARSE_ERROR"  # Default if parsing fails

    # Check if the last line matches the decision tag format
    match = re.match(r"<decision>(YES|NO)</decision>$",
                     last_line, re.IGNORECASE)
    if match:
        decision = match.group(1).upper()
        corrected_paragraph = "\n".join(lines[:-1]).strip()
        # Handle case where paragraph might be empty if API only returned the tag
        if not corrected_paragraph and len(lines) > 1:
            # Keep potential empty lines if intended
            corrected_paragraph = "\n".join(lines[:-1]).strip()
        elif not corrected_paragraph and len(lines) <= 1:
            # Add warning
            corrected_paragraph = "[Warning: Potentially missing paragraph content]"

    else:
        # Fallback: Maybe the tag is not on a new line or there's extra text.
        # Search for the tag anywhere at the end of the content.
        search_match = re.search(
            r"<decision>(YES|NO)</decision>\s*$", response_content, re.IGNORECASE | re.MULTILINE)
        if search_match:
            decision = search_match.group(1).upper()
            corrected_paragraph = response_content[:search_match.start()].strip(
            )
            print(
                f"Warning: Parsed decision tag using fallback search for content ending with: ...{response_content[-50:]}")
        else:
            # Could not find the tag reliably
            print(
                f"Warning: Could not parse decision tag from response: {response_content}")
            # Keep the full response as paragraph if tag is missing
            corrected_paragraph = response_content
            decision = "PARSE_ERROR"

    # Ensure paragraph is not empty if decision was found but split failed unexpectedly
    if decision in ["YES", "NO"] and not corrected_paragraph:
        # This might happen if the API *only* returned the tag correctly formatted
        corrected_paragraph = "[Warning: Model returned only the decision tag]"

    return corrected_paragraph, decision


def main():
    """Main function to process the CSV file."""
    print("Starting processing...")

    # 1. Load API Key
    try:
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized.")
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(
            f"An unexpected error occurred during OpenAI client initialization: {e}")
        return

    # 2. Read CSV
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Successfully read {len(df)} rows from {INPUT_CSV_FILE}.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 3. Check for required columns
    if "paragraph" not in df.columns or "question" not in df.columns:
        print("Error: CSV file must contain 'paragraph' and 'question' columns.")
        return

    # 4. Process each row
    results = []
    total_rows = len(df)
    for index, row in df.iterrows():
        print(f"\nProcessing row {index + 1}/{total_rows}...")
        original_paragraph = row.get('paragraph', '')  # Use .get for safety
        original_question = row.get('question', '')

        # Ensure data are strings and handle potential NaN/None values
        if pd.isna(original_paragraph) or pd.isna(original_question) or not original_paragraph or not original_question:
            print(
                f"Skipping row {index + 1} due to missing paragraph or question.")
            results.append({
                'original_paragraph': original_paragraph,
                'original_question': original_question,
                'corrected_paragraph': 'SKIPPED_MISSING_DATA',
                'decision': 'SKIPPED_MISSING_DATA'
            })
            continue

        paragraph_str = str(original_paragraph)
        question_str = str(original_question)

        print(f"  Question: {question_str[:100]}...")  # Print snippet
        # Print snippet
        print(f"  Original Paragraph: {paragraph_str[:100]}...")

        # Call API
        api_response_content = call_openai_api(
            client, paragraph_str, question_str)

        # Parse Response
        corrected_paragraph, decision = parse_openai_response(
            api_response_content)
        # Print snippet
        print(f"  Corrected Paragraph: {corrected_paragraph[:100]}...")
        print(f"  Decision: {decision}")

        results.append({
            'original_paragraph': original_paragraph,
            'original_question': original_question,
            'corrected_paragraph': corrected_paragraph,
            'decision': decision
        })

        # Optional: Add a small delay to avoid overwhelming the API for very large files
        # time.sleep(0.5) # Adjust delay as needed

    # 5. Save Results
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nProcessing complete. Results saved to {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"Error saving results to CSV file: {e}")


if __name__ == "__main__":
    main()
