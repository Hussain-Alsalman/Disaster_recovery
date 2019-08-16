import sys
import pickle

# import libraries
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

# To use regex
# Importing re
import re

# To Process Strings
#import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):

    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table("T_messeges", engine)
    X = df.iloc[:,0]
    Y = df.iloc[:,3:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """This function normalize the string, remove punctuations, and then create a list of tokens.
    Args:
        text: a string
    Returns:
        processed_list : a processed list of strings (normalized, punctuations free, and tokenized)
    """

    # Convert string to lowercase
    text = text.lower()

    # Remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize words
    tokenized_words = word_tokenize(text)

    stemer = SnowballStemmer("english")
    processed_list = [stemer.stem(token) for token in tokenized_words if token not in stopwords.words("english")]

    return processed_list


def build_model():

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[4,8]}

    cv = GridSearchCV(pipeline, param_grid=parameters,cv=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    pred_y  = model.predict(X= X_test)

    print(classification_report(y_pred= pred_y, y_true= Y_test.values, target_names = Y_test.columns))


def save_model(model, model_filepath):

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
