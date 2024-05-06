import os
import pandas as pd

def read_adolescent_csv(seed_directory, seed_number):
    file_name = f"adolescent#001.csv"
    file_path = os.path.join(seed_directory, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Reading {file_name} under {seed_directory} (Seed {seed_number})")
        return df
    else:
        print(f"{file_name} not found under {seed_directory} (Seed {seed_number})")
        return None

def combine_and_save_csv(base_directory, output_file="combined_adolescent.csv"):
    combined_df = pd.DataFrame()

    for seed_number in range(20):
        seed_directory = os.path.join(base_directory, f"seed{seed_number}")
        df = read_adolescent_csv(seed_directory, seed_number)

        if df is not None:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined CSV file saved as {output_file}")

def main():
    base_directory = "/home/berk/VS_Projects/simglucose/examples/T1DatasetAnalysis/PPO"
    combine_and_save_csv(base_directory)

if __name__ == "__main__":
    main()