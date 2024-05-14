#!/bin/bash

nohup python3 -u main_hnsw_fast.py --data_name wine --missing_mechanism MCAR >> ./logs/wine_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name heart --missing_mechanism MCAR >> ./logs/heart_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name breast --missing_mechanism MCAR >> ./logs/breast_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name car --missing_mechanism MCAR >> ./logs/car_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name wireless --missing_mechanism MCAR >> ./logs/wireless_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name abalone --missing_mechanism MCAR >> ./logs/abalone_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name turkiye --missing_mechanism MCAR >> ./logs/turkiye_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name letter --missing_mechanism MCAR >> ./logs/letter_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name chess --missing_mechanism MCAR >> ./logs/chess_MCAR.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name shuttle --missing_mechanism MCAR >> ./logs/shuttle_MCAR.txt 2>&1 ;