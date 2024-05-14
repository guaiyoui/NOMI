#!/bin/bash

nohup python3 -u main_miracle.py --data_name wine  >> ./logs/miracle_wine.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name heart  >> ./logs/miracle_heart.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name breast  >> ./logs/miracle_breast.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name car  >> ./logs/miracle_car.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name wireless  >> ./logs/miracle_wireless.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name abalone  >> ./logs/miracle_abalone.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name turkiye  >> ./logs/miracle_turkiye.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name letter  >> ./logs/miracle_letter.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name chess  >> ./logs/miracle_chess.txt 2>&1 &&
nohup python3 -u main_miracle.py --data_name shuttle  >> ./logs/miracle_shuttle.txt 2>&1 &