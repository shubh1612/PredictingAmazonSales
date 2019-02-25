import pandas as pd
import sklearn as sk
import numpy as np

def x_value(df, c0_cutoff, c1_cutoff, c2_cutoff, c3_cutoff):

	x = [[]]
	y = []
	x = df.iloc[:, [1, 2, 3, 4, 5, 6, 9]].values
	y = df.iloc[:, [0]].values

	y[y <= c0_cutoff] = 0
	y[y > c3_cutoff] = 4
	y[y > c2_cutoff] = 3
	y[y > c1_cutoff] = 2
	y[y > c0_cutoff] = 1
	
	return x, y

if __name__ == "__main__":
	data_dir = '../data/WithoutReviews/'
	df = pd.read_hdf(data_dir + 'No_normalized_data.h5')
	x, y = x_value(df, 20, 40, 60, 80)
	print(len(y), np.count_nonzero(y == 1), np.count_nonzero(y == 2), np.count_nonzero(y == 3), np.count_nonzero(y == 4))
