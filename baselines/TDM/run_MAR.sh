#!/bin/bash

nohup python3 -u demo.py --data_name wine --missing_mechanism MAR >> ./logs/wine.txt 2>&1 ;
nohup python3 -u demo.py --data_name heart --missing_mechanism MAR >> ./logs/heart.txt 2>&1 ;
nohup python3 -u demo.py --data_name breast --missing_mechanism MAR >> ./logs/breast.txt 2>&1 ;
nohup python3 -u demo.py --data_name car --missing_mechanism MAR >> ./logs/car.txt 2>&1 ;
nohup python3 -u demo.py --data_name wireless --missing_mechanism MAR >> ./logs/wireless.txt 2>&1 ;
nohup python3 -u demo.py --data_name abalone --missing_mechanism MAR >> ./logs/abalone.txt 2>&1 ;
nohup python3 -u demo.py --data_name turkiye --missing_mechanism MAR >> ./logs/turkiye.txt 2>&1 ;
nohup python3 -u demo.py --data_name letter --missing_mechanism MAR >> ./logs/letter.txt 2>&1 ;
nohup python3 -u demo.py --data_name chess --missing_mechanism MAR >> ./logs/chess.txt 2>&1 ;
nohup python3 -u demo.py --data_name shuttle --missing_mechanism MAR >> ./logs/shuttle.txt 2>&1 ;