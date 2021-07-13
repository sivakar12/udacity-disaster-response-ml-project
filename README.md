# Disaster Response Pipeline Project

### About
This is a project I am doing as part of the Udacity Data Scientist Nanodegree program
The core of this project is a machine learning classifier that reads messages coming during a disaster and classifies them into categories so they can be routed to the appropriate responder. It is a multilabel classification. The classification uses Bag of Words and TF-IDF techniques to model the text and there are data cleaning, tokenizing, stemming and lemmatizing tasks. The classifier is built making use of the pipeline features of Scikit Learn and Grid Search Cross Validation is used to tune the parameters
 
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
