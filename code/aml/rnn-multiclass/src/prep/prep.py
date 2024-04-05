import argparse
import pandas as pd
import re 


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

# drop duplicates
df = df.drop_duplicates(subset=["UserId","Time","Text"])

# Create columns for Positive, Negative and Neutral Reviews based on Score
df['PositiveReviews'] = df['Score'] > 3
df['NegativeReviews'] = df['Score'] < 3
df['NeutralReviews'] = df['Score'] == 3

# Summary imputation
df.loc[df.Summary.isna(), "Summary"] = "Review Summary"

# duplicated values
df_cleaned = df.drop_duplicates()

# Create a preprocessing function with optional steps
def preprocessing(sentence, remove_html=True):

    # Removing whitespaces
    sentence = sentence.strip()

    # Removing HTML tags
    if remove_html:
        sentence = re.sub(r'<.*?>', '', sentence)

    return sentence   

# apply necessary preprocessing steps
df['Text'] = df['Text'].apply(lambda x: preprocessing(x))

# change labelled data as binary for model
df['Label'] = df['PositiveReviews'].astype(int)

# save data
df_cleaned = df.to_csv(args.prep_data)