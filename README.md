# Disaster Recovery Project

## Objective 
the objective of the project is to create a machine learning solution that helps classify messages coming from different sources into pre-determine categories to expedite response time for any disasters. 

## Content 

The project consist of two notebooks that laydown the rational behind the code and the steps considered to process the data. Those files are 

  - ETL Pipeline Preparation.ipynb
  - ML Pipeline Preparation.ipynb
  
  However, the real core of the project resides in the 
  
 ### data/process_data.py
 
 This file contains all the necessary steps explained in the ETL Pipeline Preparation.ipynb file. It a pipleline that reads in the data, process it then load it on SQL database. 
 
 ### models/train_classifier.py
 
 This file contains all the ncessary steps explained in ML Pipeline Preparation.ipynb. It is a pipeline that loads the processed data and train a model then export it into a pickle file. 
 
 ### apps/run.py 
 
 This file bring all the results from the forementioned files into a neat interactive web application. It contains interesting visuals about the dataset as a whole and allow the users to type their own messages to be classified. 
