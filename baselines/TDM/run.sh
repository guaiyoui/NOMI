#!/bin/bash

nohup python3 -u demo.py --data_name wine >> ./logs/wine.txt 2>&1 ;
nohup python3 -u demo.py --data_name heart >> ./logs/heart.txt 2>&1 ;
nohup python3 -u demo.py --data_name breast >> ./logs/breast.txt 2>&1 ;
nohup python3 -u demo.py --data_name car >> ./logs/car.txt 2>&1 ;
nohup python3 -u demo.py --data_name wireless >> ./logs/wireless.txt 2>&1 ;
nohup python3 -u demo.py --data_name abalone >> ./logs/abalone.txt 2>&1 ;
nohup python3 -u demo.py --data_name turkiye >> ./logs/turkiye.txt 2>&1 ;
nohup python3 -u demo.py --data_name letter >> ./logs/letter.txt 2>&1 ;
nohup python3 -u demo.py --data_name chess >> ./logs/chess.txt 2>&1 ;
nohup python3 -u demo.py --data_name shuttle >> ./logs/shuttle.txt 2>&1 ;