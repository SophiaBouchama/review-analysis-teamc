import argparse
import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from utils import decontracted, removeNumbers, removeHtml, removePunctuations, removePatterns, removeURL

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")

args = parser.parse_args()

print("preparing data...")

lines = [f"Raw data path: {args.raw_data}", f"Data output path: {args.prep_data}"]
print(args.raw_data)
print(args.prep_data)

# reading data
reviews = pd.read_csv(args.raw_data)

# set date format to Time column
reviews.Time = reviews.Time.apply(lambda x: pd.to_datetime(x, unit='s'))

reviews['ProductId'] = reviews['ProductId'].astype('str')
reviews['UserId'] = reviews['UserId'].astype('str')
reviews['Summary'] = reviews['Summary'].astype('str')
reviews['Text'] = reviews['Text'].astype('str')

reviews.drop("ProfileName", axis=1, inplace=True)

## NLP
nltk.download('stopwords')

reviews['SentimentPolarity'] = reviews['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
reviews['Class_Labels'] = reviews['SentimentPolarity'].apply(lambda x : 1 if x == 'Positive' else 0)

reviews = reviews.drop_duplicates(subset={"UserId", "Time","Text"}, keep='first', inplace=False)

reviews=reviews[reviews.HelpfulnessNumerator <= reviews.HelpfulnessDenominator]

reviews["Sentiment"] = reviews["Score"].apply(lambda score: "positive" if score > 3 else \
                                              ("negative" if score < 3 else "not defined"))
reviews["Usefulness"] = (reviews["HelpfulnessNumerator"]/reviews["HelpfulnessDenominator"]).apply\
(lambda n: ">75%" if n > 0.75 else ("<25%" if n < 0.25 else ("25-75%" if n >= 0.25 and\
                                                                        n <= 0.75 else "useless")))


#Stemming and stopwords removal
snow = SnowballStemmer('english') #initialising the snowball stemmer
#Removing the word 'not' from stopwords
default_stopwords = set(stopwords.words('english'))
#excluding some useful words from stop words list as we doing sentiment analysis
excluding = set(['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
custom_stopwords = default_stopwords - excluding

#Store all the processed reviews
preprocessed_reviews = [] 

string = ' '    
stemed_word = ' '

for review in tqdm(reviews['Text'].values):
    filtered_sentence = []
    review = decontracted(review)
    review = removeNumbers(review)
    review = removeHtml(review)
    review = removeURL(review)
    review = removePunctuations(review)
    review = removePatterns(review)
    
    for cleaned_words in review.split():   
        if ((cleaned_words not in custom_stopwords) and (2<len(cleaned_words)<16)):
            stemed_word=(snow.stem(cleaned_words.lower()))                                   
            filtered_sentence.append(stemed_word)
        
    review = " ".join(filtered_sentence) # Final string of cleaned words    
    preprocessed_reviews.append(review.strip()) # Data corpus contaning cleaned reviews from the whole dataset

reviews['CleanedText'] = preprocessed_reviews 

df_cleaned = reviews[["ProductId", "UserId", "Time", "SentimentPolarity", "Class_Labels", "Sentiment", "Usefulness", "CleanedText"]]

# save data
df_cleaned = df_cleaned.to_csv(args.prep_data, index=False)