#!/bin/bash

nohup python3 -u main_hnsw_fast_revision.py --data_name wine  >> ./logs/timedecom_wine.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name heart  >> ./logs/timedecom_heart.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name breast  >> ./logs/timedecom_breast.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name car  >> ./logs/timedecom_car.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name wireless  >> ./logs/timedecom_wireless.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name abalone  >> ./logs/timedecom_abalone.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name turkiye  >> ./logs/timedecom_turkiye.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name letter  >> ./logs/timedecom_letter.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name chess  >> ./logs/timedecom_chess.txt 2>&1 &&
nohup python3 -u main_hnsw_fast_revision.py --data_name shuttle  >> ./logs/timedecom_shuttle.txt 2>&1 &