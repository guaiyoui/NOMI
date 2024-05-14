
nohup python3 -u main.py --data_name wine --miss_rate 0.2 >> ./logs/wine.txt 2>&1 &&
nohup python3 -u main.py --data_name poker --miss_rate 0.2 >> ./logs/large_poker.txt 2>&1 &&
nohup python3 -u main.py --data_name retail --miss_rate 0.2 >> ./logs/large_retail.txt 2>&1 &&
nohup python3 -u main.py --data_name wisdm --miss_rate 0.2 >> ./logs/large_wisdm.txt 2>&1 &&
nohup python3 -u main.py --data_name higgs --miss_rate 0.2 >> ./logs/large_higgs.txt 2>&1 &