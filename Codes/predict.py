import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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

def equalSplit(x, y, ratio):

	lenY1 = np.count_nonzero(y == 1)
	lenY2 = np.count_nonzero(y == 2)
	lenY3 = np.count_nonzero(y == 3)
	lenY4 = np.count_nonzero(y == 4)
	lenY0 = len(y) - lenY1 - lenY2 - lenY3 - lenY4

	y0 = y[y == 0]
	index = np.random.choice(y0.shape[0], min(ratio*lenY4, lenY0), replace = False)
	x0 = x[index]
	y0 = y[index]

	y1 = y[y == 1]
	index = np.random.choice(y1.shape[0], min(ratio*lenY4, lenY1), replace = False)
	x1 = x[index]
	y1 = y[index]
	
	y2 = y[y == 2]
	index = np.random.choice(y2.shape[0], min(ratio*lenY4, lenY2), replace = False)
	x2 = x[index]
	y2 = y[index]
	
	y3 = y[y == 3]
	index = np.random.choice(y3.shape[0], min(ratio*lenY4, lenY3), replace = False)
	x3 = x[index]
	y3 = y[index]

	y4 = y[y == 4]
	index = np.random.choice(y4.shape[0], min(ratio*lenY4, lenY4), replace = False)
	x4 = x[index]
	y4 = y[index]

	x = np.vstack((x0, x1, x2, x3, x4))
	y = np.vstack((y0, y1, y2, y3, y4))
	y = y.ravel()

	return x, y

if __name__ == "__main__":
	
	ratio = 6
	data_dir = '../data/WithoutReviews/'
	df = pd.read_hdf(data_dir + 'No_normalized_data.h5')

	print('x and y created')
	x, y = x_value(df, 20, 40, 60, 80)
	print('x and y equally splitted')
	x, y = equalSplit(x, y, ratio)

	print('Train and test split of x and y')
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

	print('Training started')
	clf = RandomForestClassifier()
	clf.fit(x_train, y_train)

	print('Testing Started')
	y_pred = clf.predict(x_train)
	print('Train Accuracy - ', accuracy_score(y_train, y_pred))
	y_pred = clf.predict(x_test)
	print('Test Accuracy - ', accuracy_score(y_test, y_pred))
