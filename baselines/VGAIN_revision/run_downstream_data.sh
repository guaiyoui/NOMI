#!/bin/bash

nohup python3 -u main_downstream.py --data_name wine  >> ./logs/downstream_wine.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name heart  >> ./logs/downstream_heart.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name breast  >> ./logs/downstream_breast.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name car  >> ./logs/downstream_car.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name wireless  >> ./logs/downstream_wireless.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name abalone  >> ./logs/downstream_abalone.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name turkiye  >> ./logs/downstream_turkiye.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name letter  >> ./logs/downstream_letter.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name chess  >> ./logs/downstream_chess.txt 2>&1 &&
nohup python3 -u main_downstream.py --data_name shuttle  >> ./logs/downstream_shuttle.txt 2>&1 &