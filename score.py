import pickle
import json
import pandas as pd
import joblib
from azureml.core.model import Model
import azureml.automl.core
#import xgboost
#model_path='outputs/model.pkl'
#model = joblib.load(model_path)

def init():
    global model
    
    #model_path = os.path.join(os.getenv('outputs'), 'model.pkl')
    model_path = Model.get_model_path('bestmodel')
    #model_path='outputs/model.pkl'
    model = joblib.load(model_path)



def run(data):
    try:
        data = pd.DataFrame(json.loads(data)['request'])
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error