import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

# Function to read the CSV file as df and split it into train and dev
def read_split_data(file_name, random_seed, split_ratio):
    df = pd.read_csv(file_name)
    input_folder = Path(file_name).parent
    train, dev = train_test_split(df, test_size=split_ratio, random_state=random_seed)
    # write the train and dev to the respective files
    train.to_csv(input_folder/'train_split.csv', index=False)
    dev.to_csv(input_folder/'dev_split.csv', index=False)
    print(input_folder/'dev_split.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--split_ratio', type=float, default=0.1, required=True)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    train_file = args.input_folder + '/train.csv'
    read_split_data(train_file, args.random_seed, args.split_ratio)

if __name__ == '__main__':
    main()