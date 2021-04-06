# Capstone project from Azure machine learning Nanodegree program

## Introduction

This is a capstone project from Azure machine learning nanodegree program udacity. In this project we aim to build and train the machine learning models using Hyperdrive and AutoML. Once we train our machine learning model we compare the run and pick the best one for deployment. 
The second part of this project is to deploy the model as a web service and test the functionality.





![image](https://user-images.githubusercontent.com/39105103/113671246-babac400-96d3-11eb-99f6-0a84f04dd35b.png)








For this case I have used Tic-tac-toe game states as the training dataset for my model. After training my ML model, model can be used to predict which player has won the game just by looking at the states of game.

The end goal is to understand and apply different azure machine learning services.



## Project Set Up and Installation
All the necessary dependancies will be added during runtime of the notebook. We don't need to install anything explicitly. Ofcourse, to run this project you must need python 3.6+ and AzureML SDK.

Here is how each file should be run in sequence and the precautions to be taken, to reproduce the result:
 1. First Upload all the file to azure ML studio, except screenshots part. Code has written considering you upload the dataset as well along with all the script file. If you wish to import the data from Uri then you need to modify the code accordingly.
 2. Precautions: 1st run Automl.ipynb file. Here I changed dataset slightly and uploaded it to Azure datastore. If you want to execute hyperparameter_tuning.ipynb 1st, then you need copy 1st block from azureml.ipynb.
 3. Don't completely execute all the part of the automl.ipynb script. The 2nd half of automl.ipynb contains deployment code, so before deployment, run hyperparameter_tuning.ipynb.
 4. Compare the results, and use deployement part in corresponding script.

Precautions: Make sure you have : train.py.score.py and dependancies.yaml files are in same directories as automl.ipynb.


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
Azure AutoML feature automates all the Machine learning steps and suggests the suitable model for deployment. Following is the schematics for the same.


![image](https://user-images.githubusercontent.com/39105103/113673131-36b60b80-96d6-11eb-8cff-3fa558a0576c.png)




Configuration used:

![image](https://user-images.githubusercontent.com/39105103/113679206-26edf580-96dd-11eb-8bf9-2e8ebf1f14e6.png)



This is a classification problem so I chose primary metric as 'accuracy'.
The task is not heavy so our automl experment shouldn't require more than 30-40 min, thus exp_timeout min is chosen as 30min.
As we are having 4 nodes I choose 4 max concurent iterations.
K-fold validation is 4, for 25% validation set 




### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


![image](https://user-images.githubusercontent.com/39105103/113673099-2dc53a00-96d6-11eb-9500-c12720b80923.png)




### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.


![image](https://user-images.githubusercontent.com/39105103/113675556-1e93bb80-96d9-11eb-92c0-b0b26098f77b.png)




## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
