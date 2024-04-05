# Vectorization methods for documents

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


## Document vectorizer
# vectorizer_type: tfidf if using TfidfVectorizer, doc2vec if using gensim doc2vec
# train_series, val_series, test_series: pandas series for train/validation/test with text
# params_dic: vectorizer parameters
def doc_vectorizer(train_series, val_series, test_series, vectorizer_type, params_dic):

    if vectorizer_type == "doc2vec":

        return series2doc2vec_vecs(train_series, val_series, test_series, params_dic)

    elif vectorizer_type == "tfidf":

        return series2tfidf_vecs(train_series, val_series, test_series, params_dic)
    
    else:
        
        return series2count_vecs(train_series, val_series, test_series)
    

# Function to extract vectorized train/validation/test datasets 
# from Pandas series where each element is a review string
# using CountVectroizer
def series2count_vecs(train_series, val_series, test_series):

    vect = CountVectorizer()

    X_train = vect.fit_transform(train_series)
    X_val = vect.transform(val_series)
    X_test = vect.transform(test_series)

    return X_train, X_val, X_test


# Function to extract vectorized train/validation/test datasets 
# from Pandas series where each element is a review string
# using TfidfVectorizer
def series2tfidf_vecs(train_series, val_series, test_series, params_dic):

    tfidf = TfidfVectorizer(**params_dic)

    X_train = tfidf.fit_transform(train_series)
    X_val = tfidf.transform(val_series)
    X_test = tfidf.transform(test_series)

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

    for tag_num, doc in enumerate(docs_data):
        doc2vec_data.append(create_tagged_doc(doc,tag_num))

    #doc2vec_model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
    doc2vec_model = Doc2Vec(**params_dic)
    
    doc2vec_model.build_vocab(doc2vec_data)
    doc2vec_model.train(doc2vec_data,total_examples=doc2vec_model.corpus_count,epochs=doc2vec_model.epochs)

    X_train = np.array([doc2vec_model.infer_vector(docs_train[ind].split()) for ind in range(len(docs_train))])
    X_val = np.array([doc2vec_model.infer_vector(docs_val[ind].split()) for ind in range(len(docs_val))])
    X_test = np.array([doc2vec_model.infer_vector(docs_test[ind].split()) for ind in range(len(docs_test))])
    
    return X_train, X_val, X_test













    

