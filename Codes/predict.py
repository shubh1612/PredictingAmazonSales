import pandas as pd
import sklearn as sk

def x_value(df, c0_cutoff, c1_cutoff, c2_cutoff, c3_cutoff):

	x = [[]]
	y = []

	for i in range(len(df)):
		claim = df.iloc[i]['claim']
		num_rev = df.iloc[i]['num_rev']
		avg_rat = df.iloc[i]['avg_rat']
		actualdis = df.iloc[i]['actualdis']
		dealdis = df.iloc[i]['dealdis']
		timeRem = df.iloc[i]['timeRem']
		num_type = df.iloc[i]['num_type']
		day = df.iloc[i]['day']

		x_val = [claim, num_rev, avg_rat, actualdis, dealdis, timeRem, num_type, day]
		x.append(x_val)

		if (claim <= int(c0_cutoff)):
			y.append(0)
		elif (claim <= int(c1_cutoff)):
			y.append(1)
		elif (claim <= int(c2_cutoff)):
			y.append(2)
		elif (claim <= int(c3_cutoff)):
			y.append(3)
		else:
			y.append(4)

	return x, y

if __name__ == "__main__":
	data_dir = '../data/WithoutReviews/'
	df = pd.read_hdf(data_dir + 'No_normalized_data.h5')
	x, y = x_value(df, 20, 40, 60, 80)
	print(len(x), len(y))