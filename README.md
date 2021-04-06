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

Steps to access the dataset: This repo contains 'train.csv' file, upload this to azure ML studio. You can also download it from above repo. From there the data is uploaded to azure datastore and converted to tabular form. After registering the dataset onto the datastore, it can be used in profile to train the job. 

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
AutoML tested 44 different algorithm, out of which xgboost classifier,voting ensemble and stack ensemble gave highest accuracy of 99.90 %. As I speculated before, all the features are of categorical types, tree based algorithms are outperforming others. 

As the accuracy we got is nearly 100%. I assume there is less scope of improvement. And that too can be achieved by incresing the training dataset.

The following screenshots explain the result breifly.

1. Screenshot of logs generated during the training:

![image](https://user-images.githubusercontent.com/39105103/113686539-fd38cc80-96e4-11eb-8168-8ede76d6c372.png)

This is the step where AutoML does feature engineering and prepare the training and validating sets. Moving forward, below screenshot shows the list and results of different models tested.

![image](https://user-images.githubusercontent.com/39105103/113687076-949e1f80-96e5-11eb-9e02-fa4f2ccd02d4.png)


At last  screenshot showing training is completed:

![image](https://user-images.githubusercontent.com/39105103/113687160-ada6d080-96e5-11eb-905b-5b2df1be73df.png)




2. Lets check the properties of best model. The below screenshot detects the same

![image](https://user-images.githubusercontent.com/39105103/113687364-dfb83280-96e5-11eb-94da-1619fd644b93.png)

for more details, please check the notebook.

Below screenshot shows the RunId of automl model.

![image](https://user-images.githubusercontent.com/39105103/113721087-d55d5f00-970c-11eb-8fc4-cbd27baa3a42.png)


3.Following screenshot will help us to check the performance of different model relatively.

![image](https://user-images.githubusercontent.com/39105103/113688005-81d81a80-96e6-11eb-9d05-078528ad45c6.png)

1st three models got the same validation accuracy.

## Hyperparameter Tuning

In this experiment we are going to use Azure hyperdrive. Hyperdrive is used to automate the model tuning part effectively. Here we have created another experiment for the same task but used hyperdrive instead of Automl.


![image](https://user-images.githubusercontent.com/39105103/113673099-2dc53a00-96d6-11eb-9500-c12720b80923.png)


The early termination policy and Hyperparameter sampling used is as follows:

![image](https://user-images.githubusercontent.com/39105103/113689263-c1533680-96e7-11eb-82a7-b73606479161.png)



Parameters: Learning rate is common parameter in tuning ML models. I used n_estimators to control overfitting. As it is a tree based algorithm we should consider tree depth while tuning the model.

I used Bandit policy for early stopping using a slack factor of 0.15.

Randomparametersampler is used to sample my hyperparameters. As my parameters are descrete, I can use choice method to sample the parameter randomly. This method is fast well suited for descrete dataset.



### Results

Hyperdrive sampled 12 models having different hyperparameter combination set. All these 12 models are trained and compared. For training and data cleaning, I used script named 'train.py'. 

The following screenshot shows the results we got:

1. Screenshot of RunDetails:

![image](https://user-images.githubusercontent.com/39105103/113690323-e6947480-96e8-11eb-9f44-33f01edf14a3.png)

from the above logs we can see that experiment is created and also running environment onto the cluster. After that we could see that different job IDs are sampled for tuning the hyperparameters.
From the screenshot below, we can compare the results from different hyperparameters.

![image](https://user-images.githubusercontent.com/39105103/113691566-2dcf3500-96ea-11eb-91b0-18426d88722d.png)


![image](https://user-images.githubusercontent.com/39105103/113690986-a1247700-96e9-11eb-869b-b2a6b6761342.png)

It can be seen that, the combination: Learning rate = 1.0, Tree depth  = 3 and n_estimaters = 100 got highest accuracy of 98.61%

2. Screenshot of best model trained and its properties:

![image](https://user-images.githubusercontent.com/39105103/113691624-43445f00-96ea-11eb-88c8-9d40bfc2dca0.png)

The model ID and corresponding hyperparameters are shown.



3. The below screenshot shows the code to register the best model.

![image](https://user-images.githubusercontent.com/39105103/113719799-a1356e80-970b-11eb-808e-d8bedc910786.png)




Possible Improvements: I sampled 12 combinations, We can sample more for finer tuning. We can also try deep learning models

## Model Deployment
I have deployed the model onto Azure Container Instance (ACI). Tested the endpoints from notebook itself.



![image](https://user-images.githubusercontent.com/39105103/113675556-1e93bb80-96d9-11eb-92c0-b0b26098f77b.png)


Following are the steps taken to deploy the model.
step 1: Register the best model. 

Experiment with automl gave higher accuracy than hyperparameter tuning.Hence, we used automl model for deployment. For deploying the model, 1st we need to register it onto the model datastore so that it can be called using scoring script.\

![image](https://user-images.githubusercontent.com/39105103/113694470-6e7c7d80-96ed-11eb-85ed-f197e3c558ae.png)


step 2: Configure the ACI and the interface:

![image](https://user-images.githubusercontent.com/39105103/113694670-ac79a180-96ed-11eb-8c6b-e19c35f6a8a1.png)


step 3: Deploy it !:

![image](https://user-images.githubusercontent.com/39105103/113694775-cdda8d80-96ed-11eb-94db-33955e47447e.png)


For testing purpose we send a request from the notebook and got the response as expected. Demo below explains the working of deployed model breifly.

The following screenshot shows the Active status of deployed model.

![image](https://user-images.githubusercontent.com/39105103/113695208-3fb2d700-96ee-11eb-87d5-33b17cea6c5d.png)


I have named the service as bestmodel. we can observe that deployment state is healthy. The succeeded deployment status can also be seen from below status

![image](https://user-images.githubusercontent.com/39105103/113695816-eac39080-96ee-11eb-94e0-ebd309958a6f.png)

To check the state of endpoint is active or not, I have printed the uri using endpoint service object:

![image](https://user-images.githubusercontent.com/39105103/113696071-2e1dff00-96ef-11eb-8d06-320b62814f2d.png)



step 4 How to send a query:
As model is deployed, we can consume the endpoints. 
We have created a 'service' object in notebook while deploying the model. We can fetch the service uri from this object. Note that now our model is deployed as web service so it can only be communicated by JSON request. So we created a example query, converted it to json format and then send it to the deployed service. Responce we got from the endpoints is also in json format. Below screenshot shows the code we used to send the request and get the response.


![image](https://user-images.githubusercontent.com/39105103/113727917-37b95e00-9713-11eb-9e00-78c5a46e5c32.png)





## Screen Recording

This is the screencast link, It shows the demo of working model.

Working model link:
https://drive.google.com/file/d/1voO93FuzL8_lLC3WhAS9ssW01FgAbP0g/view?usp=sharing

Video showing working demo:
https://drive.google.com/file/d/1PT7KWnhKy0WvXrS7bqJp6-3KN7iZOGv_/view?usp=sharing



**Please note that we are sending the request through service URI and not to the locally present model. This can be seen from the screencast.**
.
