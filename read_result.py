import pandas as pd
import glob
import sys

def main(folder_path):

    # Use glob to find all CSV files in the folder
    file_pattern = folder_path + '/*.csv'
    csv_files = glob.glob(file_pattern)

    # Initialize an empty list to store the dataframes
    dfs = []

    # Loop through the CSV files and read them into dataframes
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort the combined dataframe by the 'parameter' column
    sorted_df = combined_df.sort_values('parameter')

    # Print the sorted dataframe
    sorted_df.to_csv('./result.csv')


if __name__ == "__main__":

    folder_path = sys.argv[1]