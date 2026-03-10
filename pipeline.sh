#!/bin/bash

# Скрипт-конвейер для запуска всех этапов ML пайплайна
# set -e остановит скрипт, если хоть одна команда завершится с ошибкой
set -e

echo "=== Запуск ML Pipeline ==="

echo "[1/4] Запуск генерации сырых данных..."
python data_creation.py

echo "[2/4] Запуск предобработки данных (скейлинг)..."
python model_preprocessing.py

echo "[3/4] Запуск создания и обучения модели..."
python model_preparation.py

echo "[4/4] Запуск тестирования обученной модели..."
python model_testing.py

echo "=== ML Pipeline успешно завершен! ==="
