# Vectorization methods for documents

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Utils functions
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def getExistingProfileName(df, userId):
    newProfileName = "Anonymous"
    profileNames = df[df.UserId == userId].ProfileName.dropna()

    if len(profileNames):
        newProfileName = profileNames[0]

    return newProfileName

#Function to clean html tags from a sentence
def removeHtml(sentence): 
    pattern = re.compile('<.*?>')
    cleaned_text = re.sub(pattern,' ',sentence)
    return cleaned_text

#Function to keep only words containing letters A-Z and a-z. This will remove all punctuations, special characters etc. https://stackoverflow.com/a/5843547/4084039
def removePunctuations(sentence):
    cleaned_text  = re.sub('[^a-zA-Z]',' ',sentence)
    return (cleaned_text)

#Remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
def removeNumbers(sentence):
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    return (sentence)

#Remove URL from sentences.
def removeURL(sentence):
    text = re.sub(r"http\S+", " ", sentence)
    sentence = re.sub(r"www.\S+", " ", text)
    return (sentence)

#We will remove all such words which has three consecutive repeating characters.
def removePatterns(sentence): 
    cleaned_text  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',sentence)
    return (cleaned_text)

#Expand the reviews x is aninput string of any length. Convert all the words to lower case
def decontracted(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")\
                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
                           .replace("'cause'"," because")
    
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

## Document vectorizer
# vectorizer_type: tfidf if using TfidfVectorizer, doc2vec if using gensim doc2vec
# train_series, val_series, test_series: pandas series for train/validation/test with text
# params_dic: vectorizer parameters
def doc_vectorizer(train_series, val_series, test_series, vectorizer_type, params_dic):

    if vectorizer_type == "doc2vec":

        return series2doc2vec_vecs(train_series, val_series, test_series, params_dic)

    elif vectorizer_type == "tfidf":

        return series2tfidf_vecs(train_series, val_series, test_series, params_dic)
    
    elif vectorizer_type == "countVec":
        
        return series2count_vecs(train_series, val_series, test_series, params_dic)
    

# Function to extract vectorized train/validation/test datasets 
# from Pandas series where each element is a review string
# using CountVectroizer
def series2count_vecs(train_series, val_series, test_series, params_dic):

    vect = CountVectorizer(**params_dic)

    X_train = vect.fit_transform(train_series)
    X_val = vect.transform(val_series)
    X_test = vect.transform(test_series)

    return X_train, X_val, X_test, vect


# Function to extract vectorized train/validation/test datasets 
# from Pandas series where each element is a review string
# using TfidfVectorizer
def series2tfidf_vecs(train_series, val_series, test_series, params_dic):

    tfidf = TfidfVectorizer(**params_dic)

    X_train = tfidf.fit_transform(train_series)
    X_val = tfidf.transform(val_series)
    X_test = tfidf.transform(test_series)

    return X_train, X_val, X_test, tfidf

# Function to extract vectorized train/validation/test datasets 
# from Pandas series and already trained TfidfVectorizer
def extract_2tfidf_vecs(train_series, val_series, test_series, vect):

    X_train = vect.transform(train_series)
    X_val = vect.transform(val_series)
    X_test = vect.transform(test_series)

    return X_train, X_val, X_test


# Create tagged doc for doc2vec
# doc_txt: text string to use in document
# tag_number: (unique) number to associate to document 
def create_tagged_doc(doc_txt,tag_number):
    return TaggedDocument(words=doc_txt.split(), tags=[tag_number])


# Function to extract vectorized train/validation/test datasets 
# from Pandas series where each element is a review string
# using a trained doc2vec model
def series2doc2vec_vecs(train_series, val_series, test_series, params_dic):
    
    docs_train = list(train_series)
    docs_val = list(val_series)
    docs_test = list(test_series)

    docs_data = docs_train + docs_val + docs_test

    # Create labeled data for Doc2Vec training data
    doc2vec_data = []

    for tag_num, doc in enumerate(docs_train):
        doc2vec_data.append(create_tagged_doc(doc,tag_num))

    doc2vec_model = Doc2Vec(**params_dic)
    
    doc2vec_model.build_vocab(doc2vec_data)
    doc2vec_model.train(doc2vec_data,total_examples=doc2vec_model.corpus_count,epochs=doc2vec_model.epochs)

    X_train = np.array([doc2vec_model.infer_vector(docs_train[ind].split()) for ind in range(len(docs_train))])
    X_val = np.array([doc2vec_model.infer_vector(docs_val[ind].split()) for ind in range(len(docs_val))])
    X_test = np.array([doc2vec_model.infer_vector(docs_test[ind].split()) for ind in range(len(docs_test))])
    
    return X_train, X_val, X_test, doc2vec_model


# Utils functions
def getExistingProfileName(df, userId):
    newProfileName = "Anonymous"
    profileNames = df[df.UserId == userId].ProfileName.dropna()

    if len(profileNames):
        newProfileName = profileNames[0]

    return newProfileName
