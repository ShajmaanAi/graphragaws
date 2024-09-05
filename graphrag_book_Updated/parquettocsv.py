

import pandas as pd
import os


def gencsv(filename):
    output_directory='/home/devuser/Desktop/DefectAnalyser/graphrag_book_Updated/updated_csv'
    csv_filename = os.path.splitext(filename.name)[0] + '.csv'
    csv_filepath = os.path.join(output_directory, csv_filename)
    df = pd.read_parquet(filename)
    df.to_csv(csv_filepath, index=False)


# Define the directory you want to scan
directory = '/home/devuser/Desktop/DefectAnalyser/graphrag_book_Updated/output/20240820-123413/artifacts'

# Loop through all files in the specified directory (and its subdirectories)
for filename in os.scandir(directory):
    if filename.is_file() and filename.name.endswith('.parquet'):  # skip directories and other non-files
        print('Processing file:', filename.name)
        gencsv(filename)
