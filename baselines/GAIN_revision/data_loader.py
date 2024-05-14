import numpy as np
from utils import binary_sampler
from skimage import data,io
from mechanism import produce_NA

def data_loader (data_name, miss_rate, args):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['letter', 'spam', 'wine', 'heart', 'breast', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'chess', 'poker', 'abalone', 'yeast', 'car', 'shuttle', 'retail', 'wisdm', 'higgs']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    # data_x = data_x[0:4200]
    # data_x = data_x[:,0:55]
  else:
    data_x = io.imread("data/1.jpg")
  # Parameters

  feature_dim = int(data_x.shape[1]*args.feature_dim)
  data_x = data_x[:, :feature_dim]

  training_size = int(data_x.shape[0]*args.training_size)
  data_x = data_x[:training_size, :]

  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  # data_m = produce_NA(data_x, p_miss=2*miss_rate, mecha="MAR", p_obs=0.5)['mask'].numpy()
  # data_m = produce_NA(data_x, p_miss=2*miss_rate, mecha="MNAR", opt="quantile", p_obs=0.5, q=0.3)['mask'].numpy()
  miss_data_x = data_x.copy()
  if data_name == 'mnist':
    miss_data_x = miss_data_x.astype(float)
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m
