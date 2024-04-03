import argparse
import pandas as pd

from utils import getExistingProfileName

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")

args = parser.parse_args()

print("preparing data...")

lines = [f"Raw data path: {args.raw_data}", f"Data output path: {args.prep_data}"]
print(args.raw_data)
print(args.prep_data)

# reading data
df = pd.read_csv(args.raw_data)

# set date format to Time column
df.Time = df.Time.apply(lambda x: pd.to_datetime(x, unit='s'))

# Profile Name imputation
# userIdsWithoutProfileName = df[df.ProfileName.isna()].UserId
# for userId in userIdsWithoutProfileName: 
    # df.loc[df.UserId == userId, "ProfileName"] = getExistingProfileName(df, userId)

# Summary imputation
df.loc[df.Summary.isna(), "Summary"] = "Review Summary"

# duplicated values
df_cleaned = df.drop_duplicates()

# save data
df_cleaned = df_cleaned.to_csv(args.prep_data)