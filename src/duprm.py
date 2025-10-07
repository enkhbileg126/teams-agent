import pandas as pd


def remove_duplicates_pandas(input_file, output_file):
    """
    Removes duplicate rows from a CSV file using the pandas library.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)

        # Remove duplicate rows
        df_no_duplicates = df.drop_duplicates()

        # Write the DataFrame without duplicates to a new CSV file
        df_no_duplicates.to_csv(output_file, index=False)
        print(
            f"Successfully removed duplicate lines and saved the result to {output_file}"
        )

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove duplicate lines from a CSV file."
    )
    parser.add_argument("input_file", help="The input CSV file.")
    parser.add_argument("output_file", help="The output CSV file.")
    args = parser.parse_args()

    remove_duplicates_pandas(args.input_file, args.output_file)
