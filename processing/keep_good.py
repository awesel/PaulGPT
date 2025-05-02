import pandas as pd
import os

# --- Configuration ---
# Input file from the previous OpenAI processing step
INPUT_CSV_FILE = "pg_question_pairs_processed.csv"

# Output file containing only rows marked as 'YES'
OUTPUT_CSV_FILE = "pg_question_pairs_good_only.csv"

# The column name containing the 'YES'/'NO' decision
DECISION_COLUMN = "decision"

# The value indicating a "good" pair to keep
POSITIVE_DECISION_VALUE = "YES"
# ---------------------


def filter_csv_by_decision(input_path, output_path, decision_col, keep_value):
    """
    Reads a CSV file, filters rows based on a decision column,
    and saves the filtered data to a new CSV file.
    """
    print(f"Attempting to read input file: {input_path}")

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'.")
        print("Please ensure the previous script ran successfully and created this file.")
        return

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_path)
        print(f"Successfully read {len(df)} rows from {input_path}.")

    except Exception as e:
        print(f"Error reading CSV file '{input_path}': {e}")
        return

    # Check if the decision column exists
    if decision_col not in df.columns:
        print(
            f"Error: Decision column '{decision_col}' not found in the CSV file.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # --- Filtering ---
    initial_rows = len(df)
    print(f"Filtering rows where '{decision_col}' is '{keep_value}'...")

    # Filter the DataFrame - use .str.upper() for case-insensitivity just in case
    # Also handle potential non-string values if errors occurred upstream
    # The .copy() prevents potential SettingWithCopyWarning from pandas
    filtered_df = df[df[decision_col].astype(
        str).str.upper() == keep_value.upper()].copy()

    final_rows = len(filtered_df)
    removed_rows = initial_rows - final_rows

    print(f"Filtering complete:")
    print(f"  - Initial rows: {initial_rows}")
    print(f"  - Rows kept ('{decision_col}' == '{keep_value}'): {final_rows}")
    print(f"  - Rows removed: {removed_rows}")

    if final_rows == 0:
        print(
            f"Warning: No rows found where '{decision_col}' was '{keep_value}'. The output file will be empty or contain only headers.")

    # --- Save the filtered data ---
    try:
        print(f"Saving filtered data to {output_path}...")
        # index=False prevents pandas from writing the DataFrame index as a column
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully saved {final_rows} rows to {output_path}.")

    except Exception as e:
        print(f"Error saving filtered data to '{output_path}': {e}")


def main():
    """Main function to execute the filtering."""
    filter_csv_by_decision(
        input_path=INPUT_CSV_FILE,
        output_path=OUTPUT_CSV_FILE,
        decision_col=DECISION_COLUMN,
        keep_value=POSITIVE_DECISION_VALUE
    )


if __name__ == "__main__":
    main()
