#!/bin/bash

nohup python3 -u demo.py --data_name retail --missing_mechanism MCAR >> ./logs/large_retail.txt 2>&1 &&
nohup python3 -u demo.py --data_name poker --missing_mechanism MCAR >> ./logs/large_poker.txt 2>&1 &&
nohup python3 -u demo.py --data_name wisdm --missing_mechanism MCAR >> ./logs/large_wisdm.txt 2>&1 &&
nohup python3 -u demo.py --data_name higgs --missing_mechanism MCAR >> ./logs/large_higgs.txt 2>&1 &