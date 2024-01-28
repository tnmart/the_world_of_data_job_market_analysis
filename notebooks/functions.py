
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def cleaning_column_names(dataframe: pd.DataFrame) ->pd.DataFrame:
    '''
    Cleans and formats the name of the columns.
        
    Inputs:
    dataframe: Pandas DataFrame
    
    Outputs:
    dataframe: Pandas DataFrame
    '''
    dataframe2 = dataframe.copy()
    cols = []
    for col in dataframe2.columns:
        cols.append(col.lower())
    dataframe2.columns = cols
    dataframe2.columns = dataframe2.columns.str.replace(' ', '_')
    dataframe2.rename(columns={"st": "state"}, inplace=True)
    return dataframe2

def error_metrics_report(y_real_train: list, y_real_test: list, y_pred_train: list, y_pred_test: list) -> pd.DataFrame:
    '''
    Takes the predicted and real values of both a train and a test set and calculates and returns
    its various error metrics in a pandas dataframe.
    '''
    pd.set_option('display.float_format', '{:.2f}'.format)


    MAE_train = mean_absolute_error(y_real_train, y_pred_train)
    MAE_test  = mean_absolute_error(y_real_test,  y_pred_test)

    # Mean squared error
    MSE_train = mean_squared_error(y_real_train, y_pred_train)
    MSE_test  = mean_squared_error(y_real_test,  y_pred_test)

    # Root mean squared error
    RMSE_train = root_mean_squared_error(y_real_train, y_pred_train)
    RMSE_test  = root_mean_squared_error(y_real_test,  y_pred_test)

    # R2
    R2_train = round(r2_score(y_real_train, y_pred_train), 2)
    R2_test  = round(r2_score(y_real_test,  y_pred_test), 2)

    results = {"Metric": ['MAE', 'MSE', 'RMSE', 'R2'] ,
               "Train": [MAE_train, MSE_train, RMSE_train, R2_train],
               "Test":  [MAE_test, MSE_test, RMSE_test, R2_test]}

    results_df = pd.DataFrame(results)

    return results_df

def drop_columns(dataframe: pd.DataFrame, threshold:float=0.25)->list:
    nulls_percent_df = pd.DataFrame(dataframe.isna().sum()/len(dataframe)).reset_index()
    nulls_percent_df.columns = ['column_name', 'nulls_percentage']
    nulls_percent_df
    columns_above_threshold = nulls_percent_df[nulls_percent_df['nulls_percentage']>threshold]
    columns_above_threshold['column_name']
    drop_columns_list = list(columns_above_threshold['column_name'])
    return drop_columns_list

def nan_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function creates a pd.DataFrame which shows the percentage of NaN or null values for all the columns in the provided pd.DataFrame.
    '''
    missing_values_df = pd.DataFrame(round(df.isna().sum()/len(df),4)*100) 
    missing_values_df = missing_values_df.reset_index() 
    missing_values_df.columns = ['column_name', 'percentage_of_missing_values'] 
    return missing_values_df

def unique_values(dataframe: pd.DataFrame):
    '''
    Takes a dataframe and prints the unique values for every column
    '''
    for col in dataframe.columns:        
        print(f'Unique values for {col}: ')
        print(dataframe[col].unique())
        print()
    pass

def filter_outliers(dataframe:pd.DataFrame, column:str, thr=3):
    lower_limit = np.mean(dataframe[column]) - thr * np.std(dataframe[column])
    upper_limit = np.mean(dataframe[column]) + thr * np.std(dataframe[column])
    column_no_outliers = [x for x in dataframe[column] if lower_limit <= x <= upper_limit]
    return column_no_outliers