#!/bin/bash

# nohup python3 -u demo.py --data_name wine  >> ./logs/onehot_wine.txt 2>&1 &&
# nohup python3 -u demo.py --data_name heart  >> ./logs/onehot_heart.txt 2>&1 &&
# nohup python3 -u demo.py --data_name breast  >> ./logs/onehot_breast.txt 2>&1 &&
# nohup python3 -u demo.py --data_name car  >> ./logs/onehot_car.txt 2>&1 &&
# nohup python3 -u demo.py --data_name wireless  >> ./logs/onehot_wireless.txt 2>&1 &&
# nohup python3 -u demo.py --data_name abalone  >> ./logs/onehot_abalone.txt 2>&1 &&
# nohup python3 -u demo.py --data_name turkiye  >> ./logs/onehot_turkiye.txt 2>&1 &&
# nohup python3 -u demo.py --data_name letter  >> ./logs/onehot_letter.txt 2>&1 &&
nohup python3 -u demo.py --data_name car  >> ./logs/onehot_car.txt 2>&1 &&
nohup python3 -u demo.py --data_name chess  >> ./logs/onehot_chess.txt 2>&1 &&
nohup python3 -u demo.py --data_name shuttle  >> ./logs/onehot_shuttle.txt 2>&1 &