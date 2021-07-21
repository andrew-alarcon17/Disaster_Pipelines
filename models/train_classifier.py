import sys
import os
import re
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlite3 import connect

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """
    Loads the data
    
    Takes In:
        database_filepath: SQLite Path
    Output:
        X: features for our model
        Y: predictor
        y_labels: names of the columns in the Y DataFrame
        
    """
    
    engine = create_engine('sqlite:///' + (database_filepath))
    df = pd.read_sql_table('df', engine)
    
    #df['related'] = df['related'].replace(2, 1)
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    df = df.drop(['child_alone'], axis=1)
    
    X = df.message
    y = df.iloc[:, 4:]

    y_labels = y.columns
    
    return X, y, y_labels

def tokenize(text):
    """
    Tokenization function to process the text data.
    
    Takes In:
        text: Text message that needs to be tokenized
    Outputs:
        clean_tokens: List made from each tokenized text
    """
    
    #make the text lowercase
    text = text.lower()
    
    #remove punctuation from the text
    text = text.replace(r'[^\w\s]','')
    
    #stop_words_ = set(stopwords.words('english'))
    
    #tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        #lemmatize, normalize case, remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Takes in the starting verb of a sentence to be used as a feature
    for the classifier.
    """
    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':

                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)
    

def build_model():
    """
    Takes in the message column as input and output classification results 
    on the other 36 categories in the dataset.
    
    Output:
        ML pipeline that classifies text messages
    """

    parameters = [
    {
        'clf__estimator__n_estimators': [10, 20],
        'vect__max_df': (0.5, 1.0),
        #'clf__estimator__criterion': ('gini', 'entropy'),
        #'clf__estimator__min_samples_split': (2, 3)
    }
]
    """
    pipeline = Pipeline([
        
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf_transformer', TfidfTransformer())
                ])),

                ('starting_verb_transformer', StartingVerbExtractor())
            ])),
        
            ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    This pipeline would be used without the implementation of GridSearch
    """
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=4, verbose=2)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model by applying it to the test set and measures the performance.
    
    Takes In:
        model: The pipeline made from build_model
        X_test: Testing features
        Y_test: Testing predictor
        category_names: y_labels
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_pred, target_names=Y_test.columns.values))
    #print(classification_report(Y_test.values, Y_pred, target_names=Y_test.columns))


def save_model(model, model_filepath):
    """
    Saves the trained model as a Pickle file.
    
    Takes In:
        model: the trained model
        model_filepath: where the Pickle file will be saved.
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()