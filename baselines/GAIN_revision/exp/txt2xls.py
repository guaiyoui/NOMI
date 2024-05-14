import numpy as np
import pandas as pd

df = pd.read_table('test.txt', sep=' ', header=None)

df.to_excel('excel_output.xls',na_rep=11,index=False,header=None)