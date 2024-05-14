#!/bin/bash

nohup python3 -u main_downstream.py --data_name wine  >> ./logs/downstream.py_wine.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name heart  >> ./logs/downstream.py_heart.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name breast  >> ./logs/downstream.py_breast.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name car  >> ./logs/downstream.py_car.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name wireless  >> ./logs/downstream.py_wireless.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name abalone  >> ./logs/downstream.py_abalone.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name turkiye  >> ./logs/downstream.py_turkiye.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name letter  >> ./logs/downstream.py_letter.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name chess  >> ./logs/downstream.py_chess.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name shuttle  >> ./logs/downstream.py_shuttle.txt 2>&1 &