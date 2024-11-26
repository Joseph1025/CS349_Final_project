from read import *
from regression import *


def main():
    # Extract the data from the Excel file
    extract_data("./data/StockX-Data-Contest-2019-3.xlsx")

    # Split the data into training, testing, and validation sets
    train_data, test_data, valid_data = split_data('./data/raw_data.csv')

    print("Data split into training, testing, and validation sets.")
    print("They are saved in the data folder.")

    # Perform regression on the data
    scikit_regression(train_data, test_data, valid_data)


if __name__ == "__main__":
    main()