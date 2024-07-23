from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()

def convert_to_camel(s: str) -> str:
    """
    Converts a string to camel case.
    
    Args:
        s (str): The input string to be converted.
        
    Returns:
        str: The converted string in camel case.
    """
    words = s.split("_")
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

@app.command()
def process_data(input_path: Path, output_path: Path):
    """
    Process the data by adding SUBJID and VISIT columns and converting column names to camel case.

    Args:
        input_path (Path): The path to the input CSV file.
        output_path (Path): The path to save the output CSV file.
    """
    df = pd.read_csv(input_path, sep=";", encoding="latin1")

    # Add a SUBJID column at the first position
    df.insert(0, "SUBJID", range(1, 1 + len(df)))

    # Add a VISIT column at the second position
    df.insert(1, "VISIT", 1)

    # Convert the dataframe column names to camel case
    df.columns = [convert_to_camel(col) if col not in ["SUBJID", "VISIT"] else col for col in df.columns]

    # Save the processed dataframe to a new CSV file
    df.to_csv(output_path, index=False)

    # Print the first few rows of the processed dataframe
    print(df.head())

    # Print all column names of the processed dataframe
    print(df.columns)

if __name__ == "__main__":
    app()
