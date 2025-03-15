import pandas as pd
import argparse
from pathlib import Path

def sample_csv_rows(input_file: str, num_rows: int, output_file: str = None, seed: int = None) -> pd.DataFrame:
    """
    Sample random rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        num_rows (int): Number of rows to sample
        output_file (str, optional): Path to save sampled rows
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Sampled rows
    """
    # Check if file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    try:
        # Read CSV
        df = pd.read_csv(input_file)
        
        # Check if requested rows exceed total rows
        total_rows = len(df)
        if num_rows > total_rows:
            print(f"Warning: Requested {num_rows} rows but file only has {total_rows} rows.")
            num_rows = total_rows
        
        # Sample rows
        sampled_df = df.sample(n=num_rows, random_state=seed)
        
        # Save to file if output path is provided
        if output_file:
            sampled_df.to_csv(output_file, index=False)
            print(f"Sampled rows saved to: {output_file}")
            
        return sampled_df
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Sample random rows from a CSV file')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('num_rows', type=int, help='Number of rows to sample')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--seed', '-s', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        sampled_df = sample_csv_rows(args.input_file, args.num_rows, args.output, args.seed)
        if sampled_df is not None:
            print("\nSampled rows:")
            print(sampled_df)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()