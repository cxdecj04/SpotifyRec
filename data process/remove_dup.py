import pandas as pd
import os

def remove_track_duplicates_from_csv(input_file, output_file=None, keep='first'):
    """
    Remove duplicate rows from a CSV file based on track_name column.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file. If None, will append '_no_duplicates' to input filename
    keep (str): Which duplicate to keep {'first', 'last', False}, default 'first'
                - 'first': Keep first occurrence
                - 'last': Keep last occurrence
                - False: Drop all duplicates
    
    Returns:
    str: Path to the output CSV file
    """
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_new{file_ext}"
    
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Validate that track_name exists in the DataFrame
    if 'track_name' not in df.columns:
        raise ValueError("Column 'track_name' not found in CSV file")
    
    # Remove duplicates and reset index
    df_cleaned = df.drop_duplicates(subset='track_name', keep=keep).reset_index(drop=True)
    
    # Save cleaned DataFrame to CSV
    try:
        df_cleaned.to_csv(output_file, index=False)
    except Exception as e:
        raise Exception(f"Error writing to output file: {str(e)}")
    
    # Print summary of cleaning
    n_removed = len(df) - len(df_cleaned)
    print(f"Removed {n_removed} duplicate rows")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_cleaned.shape}")
    print(f"Cleaned data saved to: {output_file}")
    
    return output_file

# Example usage:
if __name__ == "__main__":
    # Example with sample CSV file
    try:
        # You can replace this with your actual CSV file path
        input_csv = "dataset.csv"
        
        # Remove duplicates and save to new file
        output_csv = remove_track_duplicates_from_csv(
            input_file=input_csv,
            keep='first'  # or 'last' or False
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")