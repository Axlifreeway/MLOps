#!/bin/bash
set -e

echo "Установка зависимостей"
pip install -r requirements.txt

echo "Запуск генерации сырых данных"
python data_creation.py

echo "Запуск предобработки данных (скейлинг)"
python model_preprocessing.py

echo "Запуск обучения модели"
python model_preparation.py

echo "Запуск тестирования обученной модели"
python model_testing.py
