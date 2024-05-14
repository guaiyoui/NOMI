nohup python3 -u main_hnsw_fast.py --data_name letter --max_iter 1 >> ./logs/letter_iteration_1.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name chess --max_iter 1 >> ./logs/chess_iteration_1.txt 2>&1 ;
nohup python3 -u main_hnsw_fast.py --data_name shuttle --max_iter 1 >> ./logs/shuttle_iteration_1.txt 2>&1 ;