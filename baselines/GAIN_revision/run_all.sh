
nohup python3 -u main.py --data_name wine --miss_rate 0.2 >> ./logs/wine.txt 2>&1 ;
nohup python3 -u main.py --data_name heart --miss_rate 0.2 >> ./logs/heart.txt 2>&1 ;
nohup python3 -u main.py --data_name breast --miss_rate 0.2 >> ./logs/breast.txt 2>&1 ;
nohup python3 -u main.py --data_name car --miss_rate 0.2 >> ./logs/car.txt 2>&1 ;
nohup python3 -u main.py --data_name wireless --miss_rate 0.2 >> ./logs/wireless.txt 2>&1 ;
nohup python3 -u main.py --data_name abalone --miss_rate 0.2 >> ./logs/abalone.txt 2>&1 ;
nohup python3 -u main.py --data_name turkiye --miss_rate 0.2 >> ./logs/turkiye.txt 2>&1 ;
nohup python3 -u main.py --data_name poker --miss_rate 0.2 >> ./logs/poker.txt 2>&1 ;
nohup python3 -u main.py --data_name retail --miss_rate 0.2 >> ./logs/retail.txt 2>&1 ;

CUDA_VISIBLE_DEVICES=-1 python3 -u main.py --data_name poker --miss_rate 0.2 >> ./logs/poker.txt 2>&1