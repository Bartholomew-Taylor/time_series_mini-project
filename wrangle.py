import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
np.random.seed(99)
from sklearn.metrics import mean_squared_error
from math import sqrt



def get_iceland():
    filename = 'iceland_temps.csv'
    
    if os.path.exists(filename):
        
        iceland_df = pd.read_csv(filename)
        return iceland_df
    
    else:
        df_whole = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
        iceland_df = df_whole[df_whole['Country'] == 'Iceland']
        iceland_df.to_csv('iceland_temps.csv', index=False)
        
    return iceland_df



def prep_iceland(df):
    df = df.fillna(1)
    df['dt'] = pd.to_datetime(df['dt'], infer_datetime_format = True)
    df = df.set_index('dt')
    df = df.sort_index()
    df_resamp = df.drop(columns = ['AverageTemperatureUncertainty', 'Country'])
    return df_resamp


def split_time(df):
    train_size = int(round(df.shape[0] * 0.5))
    validate_size = int(round(df.shape[0] * 0.3))
    test_size = int(round(df.shape[0] * 0.2))
    
    validate_end_index = train_size + validate_size
    
    train = df[:train_size]
    validate = df[train_size:validate_end_index]
    test = df[validate_end_index:]
    
    return train, validate, test