import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.startswith('train_data_') or f.startswith('test_data_'):
            path = os.path.join(folder, f)
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)

def main():
    
    if not os.path.exists('train') or not os.path.exists('test'):
        print("Папки с данными не найдены!")
        return

    train_data = load_data('train')
    test_data = load_data('test')

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[['y']])
    train_data['y'] = scaled_train

    scaled_test = scaler.transform(test_data[['y']])
    test_data['y'] = scaled_test

    train_data.to_csv('train/train_preprocessed.csv', index=False)
    test_data.to_csv('test/test_preprocessed.csv', index=False)
    
    print("Данные сохранены")

if __name__ == "__main__":
    main()
