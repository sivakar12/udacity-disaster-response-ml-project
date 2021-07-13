# Disaster Response Pipeline Project

### Motivation
This is a project I am doing as part of the Udacity Data Scientist Nanodegree program
The core of this project is a machine learning classifier that reads messages coming during a disaster and classifies them into categories so they can be routed to the appropriate responder. 

### About
It is a multilabel classification. The classification uses Bag of Words and TF-IDF techniques to model the text and there are data cleaning, tokenizing, stemming and lemmatizing tasks. The classifier is built making use of the pipeline features of Scikit Learn and Grid Search Cross Validation is used to tune the parameters
 

### File Descriptions
1. **data/process_data.py** - The ETL script
2. **models/train_classifier** - The model building steps
3. **app/run.py** - The server for the website
4. **app/templates** - The website HTML files
5. **data/*.csv** - The dataset

### Installation
1. Run `pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
1. Udacity Data Scientist Nanodegree Program