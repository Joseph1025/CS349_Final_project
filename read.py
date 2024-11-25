"""
Read in an xlsx file and return a pandas dataframe.
Seperate the data into train, test, and validation sets.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_data(file_path):
    """
    Read in an xlsx file and return a pandas dataframe.
    """
    # Check if the file is read or not
    if 'raw_data.csv' in os.listdir('./data'):
        print("The data has been extracted.")
        return

    # Read the Excel file
    excel_data = pd.ExcelFile(file_path)

    # Iterate through each sheet and save it as a separate file
    for sheet_name in excel_data.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        sheet_name = sheet_name.replace(" ", "_").lower()

        # Generate a new file name for the sheet
        # The new files should also locate in the data folder
        new_file_name = f"./data/{sheet_name}.csv"

        # Save the DataFrame as a new csv file
        df.to_csv(new_file_name, index=False)
        print(f"Saved {sheet_name} to {new_file_name}")


def split_data(data, train_size=0.7, test_size=0.15, val_size=0.15):
    """
    Split the data into training, testing, and validation sets.
    Save the data as a pandas DataFrame, and return the DataFrames, in the order of train, test, and validation.
    """
    # Validate that the sizes sum to 1
    total_size = train_size + test_size + val_size
    if not (0.99 <= total_size <= 1.01):
        raise ValueError("train_size, test_size, and val_size must sum to 1.")

    data = pd.read_csv(data)
    # Some basic preprocess
    data = data.dropna()
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Release Date'] = pd.to_datetime(data['Release Date'])
    data['Days Since Release'] = (data['Order Date'] - data['Release Date']).dt.days

    # First split: Train and remaining (test + validation)
    train_data, remaining_data = train_test_split(data, train_size=train_size, random_state=42)

    # Second split: Test and validation
    test_ratio = test_size / (test_size + val_size)
    test_data, val_data = train_test_split(remaining_data, train_size=test_ratio, random_state=42)

    # Save as DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)

    # Save to CSV (optional)
    train_df.to_csv("./data/train_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)
    val_df.to_csv("./data/val_data.csv", index=False)

    return train_df, test_df, val_df


# if __name__ == "__main__":
#     extract_data("./data/StockX-Data-Contest-2019-3.xlsx")
#     split_data('./data/raw_data.csv')