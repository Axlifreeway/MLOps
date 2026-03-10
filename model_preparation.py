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
            if 'preprocessed' in filename:
                dfs.append(pd.read_csv(os.path.join(dirname, filename)))
    
    if not dfs:
        raise ValueError("DATA_LOAD_ERROR: No data found")
    
    df = pd.concat(dfs, ignore_index=True)
    x, y = df.drop('y', axis=1), df['y']
    
    return x, y

def create_and_fit(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=None,
    criterion='squared_error',
    min_samples_split=2,
    random_state=42
):
    base_model = RandomForestRegressor(random_state=random_state)
    
    distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 60],
        'criterion': ['squared_error'],
        'min_samples_split': [2, 4, 10],
    }
    
    search = RandomizedSearchCV(
        base_model, 
        distributions, 
        random_state=random_state,
        n_iter=10, 
        cv=3,      
        scoring='r2'
    )
    
    try:
        search.fit(X_train, y_train)
        print("Модель успешно обучена. Лучшие параметры:", search.best_params_)
        return search.best_estimator_
    except Exception as e:
        raise RuntimeError(f"MODEL_FIT_ERROR: {e}")

def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        print(f"Модель сохранена в {filename}")
    except Exception as e:
        raise IOError(f"MODEL_SAVE_ERROR: {e}")

def main():
    X_train, y_train = load_data(TRAIN_DIR)
    
    model = create_and_fit(X_train, y_train)

    save_model(model, MODEL_NAME)

if __name__ == "__main__":
    main()