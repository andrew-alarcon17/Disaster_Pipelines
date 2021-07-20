# Disaster_Pipelines

## You can view the webpage version of this project via this [Heroku Link](https://disaster-pipeline-app17.herokuapp.com/)

### Motivation

Messages involving disasters were used in a machine learning pipeline in order to classify whether they involved occurences such as storms, hurricanes, transportation, etc. On the web app, a message can be inputed in the navigation bar in order to classify which of the 36 categories the message falls under.

### Data Description
Disaster related messages were provided by Figure Eight and Udacity.

### Folder Descriptions

#### App Folder
Contains the html templates and run.py files to display correctly for the heroku web app.

#### Data Folder
Contains the csv files where the data is stored. The process.py file cleans up and pre processes the data and is then stored in a SQLite database.

#### Models Folder
Contains the train_classifer.py file which builds, trains, evaluates, and saves the model to a pkl file.

*Note that the jupyter files in the data and model folders are included to better understand the direction of the ETL and ML sections of this project.*

### Instructions for Running the Project Locally
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgements
The data used in this project was provided by Figure Eight.

### Webpage Preview

<img src="https://github.com/andrew-alarcon17/Disaster_Pipelines/blob/main/Images/Ave_Message_Length.png" width="500">

<img src="https://github.com/andrew-alarcon17/Disaster_Pipelines/blob/main/Images/Categories_Count.png" width="500">

<img src="https://github.com/andrew-alarcon17/Disaster_Pipelines/blob/main/Images/Message_Genres.png" width="500">



