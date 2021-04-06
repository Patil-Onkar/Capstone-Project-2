# Capstone project from Azure machine learning Nanodegree program

## Introduction

This is a capstone project from Azure machine learning nanodegree program udacity. In this project we aim to build and train the machine learning models using Hyperdrive and AutoML. Once we train our machine learning model we compare the run and pick the best one for deployment. 
The second part of this project is to deploy the model as a web service and test the functionality.

For this case I have used Tic-tac-toe game states as the training dataset for my model. After training my ML model, model can be used to predict which player has won the game just by looking at the states of game.

The end goal is to understand different azure machine learning services.



## Project Set Up and Installation
All the necessary dependancies will be added during runtime of the notebook. We don't need to install anything explicitly. Ofcourse, to run this project you must need python 3.6+ and AzureML SDK.

## Dataset

### Overview
The dataset is about states of tic-tac-toe game. In this dataset player 'x' and player 'o' are playing. Target column predicts if player 'x is winning os losing.

Below screenshot shows how the game looks and winning condition of game.

![image](https://user-images.githubusercontent.com/39105103/113667701-c952ac80-96ce-11eb-97e6-e51825a8f4fa.png)


This is the dataset screenshot. 9 columns are for 9 boxes in the game. x= player x, o=player 'o'  and b=blank

![image](https://user-images.githubusercontent.com/39105103/113668333-b4c2e400-96cf-11eb-9b8d-f9023b77231f.png)


positive: x win, Negative : x loose


The dataset is available at open UCI repo : https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame 

### Task

From this dataset, After we train our model, we want predict if player x wins or not at any state of game. Though this task can be hard coded with the rules of game, its fascinating to see how ML infers the rules outof it and predict the winning condition. Feature selection is straight forward, we use occupation state of each block. So for 9 blocks we have 9 features to train.

### Access
You can directly download the data file from the repo https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame. Its opensource, we don't require any access/authentication procedure. Alternatively I have downloaded the data and added column names, that you can find it in this repo. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
