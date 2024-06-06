#!/bin/bash

nohup python3 -u demo.py --data_name wine --missing_mechanism MNARL >> ./logs/wine.txt 2>&1 ;
nohup python3 -u demo.py --data_name heart --missing_mechanism MNARL >> ./logs/heart.txt 2>&1 ;
nohup python3 -u demo.py --data_name breast --missing_mechanism MNARL >> ./logs/breast.txt 2>&1 ;
nohup python3 -u demo.py --data_name car --missing_mechanism MNARL >> ./logs/car.txt 2>&1 ;
nohup python3 -u demo.py --data_name wireless --missing_mechanism MNARL >> ./logs/wireless.txt 2>&1 ;
nohup python3 -u demo.py --data_name abalone --missing_mechanism MNARL >> ./logs/abalone.txt 2>&1 ;
nohup python3 -u demo.py --data_name turkiye --missing_mechanism MNARL >> ./logs/turkiye.txt 2>&1 ;
nohup python3 -u demo.py --data_name letter --missing_mechanism MNARL >> ./logs/letter.txt 2>&1 ;
nohup python3 -u demo.py --data_name chess --missing_mechanism MNARL >> ./logs/chess.txt 2>&1 ;
nohup python3 -u demo.py --data_name shuttle --missing_mechanism MNARL >> ./logs/shuttle.txt 2>&1 ;