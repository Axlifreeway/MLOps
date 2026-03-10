import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

DATA_DIR = './' 
TRAIN = 'train/'
TRAIN_DIR = DATA_DIR + TRAIN
MODEL_NAME = 'giga_forest_model.joblib'

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

def create_and_fit(
    x_train,
    y_train,
    n_estimators=200,
    max_depth=None,
    criterion='mse',
    min_samples_split=2,
    random_state=42
):
    distributions = {
        n_estimators: [100, 200, 300],
        max_depth: [None, 10, 20, 60],
        criterion: ['mse'],
        min_samples_split: [2, 4, 10],
    }
 
    try:       
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        
        clf = RandomizedSearchCV(model, distributions, random_state=random_state)        
    except Exception as ex:
        print(400, 'MODEL_CREATION_ERROR')    
        
    try:
        clf.fit(x_train, y_train)
    except Exception as ex:
        print(400, 'MODEL_FIT_ERROR') 
        
    return clf.best_estimator_

def save_model(model, filename):
    try:
        joblib.dump(model, filename)
    except Exception as ex:
        print(400, "MODEL_SAVE_ERROR")
        
x_data, y_data = load_data(TRAIN_DIR)

model = create_and_fit(
    x_train=x_data,
    y_train=y_data
)

print("Модель успешно обучена!")

save_model(model, MODEL_NAME)

print(f"Модель сохранена по пути {MODEL_NAME}")