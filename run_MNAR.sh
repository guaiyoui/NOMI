#!/bin/bash

nohup python3 -u main_hnsw_fast.py --data_name wine --missing_mechanism MNAR >> ./logs/wine_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name heart --missing_mechanism MNAR >> ./logs/heart_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name breast --missing_mechanism MNAR >> ./logs/breast_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name car --missing_mechanism MNAR >> ./logs/car_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name wireless --missing_mechanism MNAR >> ./logs/wireless_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name abalone --missing_mechanism MNAR >> ./logs/abalone_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name turkiye --missing_mechanism MNAR >> ./logs/turkiye_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name letter --missing_mechanism MNAR >> ./logs/letter_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name chess --missing_mechanism MNAR >> ./logs/chess_MNAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name shuttle --missing_mechanism MNAR >> ./logs/shuttle_MNAR.txt 2>&1 ;