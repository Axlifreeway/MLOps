import os
import random
import numpy as np
import pandas as pd

def generate_data(num_samples=1000):
    x = np.linspace(0, 50, num_samples)
    y = np.sin(x) + np.random.normal(0, 0.2, num_samples)
    
    for _ in range(15):
        idx = random.randint(0, num_samples - 1)
        y[idx] += random.choice([-3, 3, -4, 4])
        
    return pd.DataFrame({'time': x, 'temperature': y})

def main():
    if not os.path.exists('train'):
        os.mkdir('train')
    
    if not os.path.exists('test'):
        os.mkdir('test')
        
    for i in range(4):
        df = generate_data(1000)
        df.to_csv(f'train/train_data_{i}.csv', index=False)
        
    for i in range(2):
        df = generate_data(500)
        df.to_csv(f'test/test_data_{i}.csv', index=False)
        
    print("Данные успешно созданы!")

if __name__ == "__main__":
    main()
