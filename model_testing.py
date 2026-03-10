import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import numpy as np

DATA_DIR = './' 
TEST = 'test/'
TEST_DIR = DATA_DIR + TEST
MODEL_NAME = './giga_forest_model.joblib'

def load_data(path):
    dfs = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            dfs.append(pd.read_csv(os.path.join(dirname, filename)))
    
    if not dfs:
        print(404, "DATA_LOAD_ERROR")
        raise ValueError()
    
    df = pd.concat(dfs, ignore_index=True)
    x, y = df.drop('y', axis=1), df['y']
    
    return x, y

def load_model(path):
    try:
        loaded_model = joblib.load(path)
        return loaded_model
    except Exception as ex:
        print(404, 'MODEL_LOAD_ERROR')
        raise

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    r2 = model.score(x_test, y_test)                
    mae = mean_absolute_error(y_test, y_pred)       
    mse = mean_squared_error(y_test, y_pred)        
    rmse = np.sqrt(mse)
    
    return {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def main():
    x_data, y_data = load_data(TEST_DIR)

    model = load_model(MODEL_NAME)

    metrics = evaluate_model(model, x_data, y_data)

    print("Метрики регрессии на тестовых данных:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        
if __name__ == "__main__":
    main()