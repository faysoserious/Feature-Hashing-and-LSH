# -*- coding: utf-8 -*-


import json
import glob
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def body_to_words( raw_text ):
    # Function to convert a raw body to a string of words
    # The input is a single string (a raw article body), and 
    # the output is a single string (a preprocessed article body)
    #
    # 1. Remove HTML
    review_body = BeautifulSoup(raw_text['body'],"lxml").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_body) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                         
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


def feature_hashing_encoding(hashing_features, vocab):
    #Create a hashing table
    hash_value= np.zeros((10377,1000))
    features = [[] for _ in range(1000)]
    value = [[] for _ in range(1000)]
    # Sum up the counts of each vocabulary word
    for i in range(len(vocab)):
        features[i%1000].append(vocab[i])
        value[i%1000].append(hashing_features[:,i])
    for j in range(1000):
        hash_value[:,j]=sum(value[j])
    return hash_value

def random_forest(features):
    from sklearn.model_selection import train_test_split
    #Use 50 trees (n_estimators) in your classifier.
    num_test = 0.20
    #Use 80% of the data for training data and 20% for test data. 
    X_train, X_test, y_train, y_test = train_test_split(train_data_features,reuter_topic, test_size=num_test, random_state=23)
    print ("Training the random forest...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 50) 
    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( X_train, y_train ) 
    result = forest.predict(X_test)
    from sklearn.metrics import make_scorer, accuracy_score
    fraction=accuracy_score(y_test, result)
    return fraction    
    
     
articles=list()
reuters = list()
reuter_in_words = list()
reuter_topic = list()
vocab = list()

path = glob.glob("D:\\New folder (2)\\2017Aug_Exam\\reuters-21578-json-master\\reuters-21578-json-master\\data\\full\\*.json")
for filename in path:
    with open(filename) as data:
            articles=(json.load(data))
            for document_list in articles:
                if (('body' in document_list)and('topics' in document_list)):
                    reuters.append(document_list)
                 
for each_reuter in reuters:
    reuter_in_words.append(body_to_words( each_reuter ))
    if ('earn' in each_reuter['topics']):
        reuter_topic.append(1)
    else:
        reuter_topic.append(0)
vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = None) 
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
train_data_features = vectorizer.fit_transform(reuter_in_words) 
fraction_raw_BOW = random_forest(train_data_features)

hashing_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names() 
hashing_table = feature_hashing_encoding(hashing_features, vocab)
fraction_hash = random_forest(hashing_table)     

