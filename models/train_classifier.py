import sys
import re
import pickle
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    all_columns = df.columns
    Y_columns = all_columns[4:]
    X = df['message']
    Y = df[Y_columns]
    Y.drop(columns=['child_alone'], inplace=True)
    return X, Y, Y_columns

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords_list = stopwords.words('english')

def tokenize(text):
    text = re.sub(r'^says: ', '', text)
    text = text.lower()
    text = re.sub(r'[^A-Za-z]', ' ', text)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens if token not in stopwords_list]
    return tokens


def build_model():
    model = Pipeline([
        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('cls', MultiOutputClassifier(LogisticRegression(), n_jobs=-1))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    for category in Y_test.columns:
        print('-----{}------'.format(category.upper()))
        print(classification_report(Y_test[category], Y_pred[category]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


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