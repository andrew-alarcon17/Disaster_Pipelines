import json
import plotly
import pandas as pd
import nltk
import sys
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as goplot
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

#References the pkl file that the train_classier makes

app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    
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
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine('sqlite:///../data/YourDatabaseName.db')
engine = create_engine('sqlite:///../data/DisasterResponse.db')
#df = pd.read_sql_table('YourTableName', engine)
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    Y = df.iloc[:,4:]

    category_counts = list(Y.sum(axis = 0).values)
    category_names = list(Y.columns)
    category_boolean = (df.iloc[:,4:] != 0).sum().values

    fig = px.pie(df, values=genre_counts, names=genre_names, title='Message Genre Distribution')
    #fig2 = px.bar(df, x=category_names, y=category_counts, title='Category Distribution' )

    #Graph 3
    df['message_length'] = df['message'].apply(len)
    genre_mean = df.groupby('genre', as_index=False)['message_length'].mean()


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Graph 1 - Average message length by genre
        {
            'data':[
                Bar(
                    x=genre_mean.genre,
                    y=genre_mean.message_length
                )
            ],

            'layout': {
                'title': 'Average Message Length by Genre',
                'yaxis': {
                    'title': 'Average Message Length'
                },
                'xaxis': {
                    'title': 'Genre'
                }
            }
        },
        #Graph 2
        {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': 'Count of Categories'
                },
                'xaxis': {
                    'title': 'Category Name'
                }
            }
        },
        #Graph 3
        fig

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

    


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    '''
    as the main function this runs whenever the file is called
    
    it sets the port and then runs the app through the desired port
    '''
    
    if len(sys.argv) == 2:
        #for some reason, changing this to 4 messes up graph 2
        app.run(host='0.0.0.0', port=int(sys.argv[1]), debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
  


if __name__ == '__main__':
    main()