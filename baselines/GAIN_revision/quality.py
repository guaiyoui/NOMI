from skimage import data,io
from utils import binary_sampler


original = io.imread("data/1.jpg")

no, dim = original.shape

miss_rate = 0.5

data_m = binary_sampler(1-miss_rate, no, dim)

i_name = "2.jpg"
io.imsave(i_name, data_m)