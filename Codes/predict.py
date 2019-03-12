import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def x_value(df, c0_cutoff, c1_cutoff, c2_cutoff, c3_cutoff):

	x = [[]]
	y = []
	x = df.iloc[:, [1, 2, 3, 4, 6, \
	 	10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 \
	 	]].values
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

	y0 = y[np.where(y == 0), :][0]
	x0 = x[np.where(y == 0), :][0]
	index = np.random.choice(y0.shape[0], min(ratio*lenY4, lenY0), replace = False)
	x0 = x0[index]
	y0 = y0[index]

	y1 = y[np.where(y == 1), :][0]
	x1 = x[np.where(y == 1), :][0]
	index = np.random.choice(y1.shape[0], min(ratio*lenY4, lenY1), replace = False)
	x1 = x1[index]
	y1 = y1[index]
	
	y2 = y[np.where(y == 2), :][0]
	x2 = x[np.where(y == 2), :][0]
	index = np.random.choice(y2.shape[0], min(ratio*lenY4, lenY2), replace = False)
	x2 = x2[index]
	y2 = y2[index]
	
	y3 = y[np.where(y == 3), :][0]
	x3 = x[np.where(y == 3), :][0]
	index = np.random.choice(y3.shape[0], min(ratio*lenY4, lenY3), replace = False)
	x3 = x3[index]
	y3 = y3[index]

	y4 = y[np.where(y == 4), :][0]
	x4 = x[np.where(y == 4), :][0]
	index = np.random.choice(y4.shape[0], min(ratio*lenY4, lenY4), replace = False)
	x4 = x4[index]
	y4 = y4[index]

	x = np.vstack((x0, x1))
	x = np.vstack((x, x2))
	x = np.vstack((x, x3))
	x = np.vstack((x, x4))
	y = np.vstack((y0, y1))
	y = np.vstack((y, y2))
	y = np.vstack((y, y3))
	y = np.vstack((y, y4))

	return x, y


def model(numFeatures):

	model = Sequential()
	model.add(Dense(12, input_dim = numFeatures, activation = 'relu'))
	model.add(Dense(8, activation = 'relu'))
	model.add(Dense(5, activation = 'softmax'))
	

def run_model(Loss, optimizer):

	print(model.summary())
	model.compile(loss = Loss, optimizer = optimizer, metrics = ['accuracy'])
	earlystop = EarlyStopping(monitor = 'val_loss', patience = PAT)
    check_pt = ModelCheckpoint(base_dir + 'model_.h5', save_best_only=True)
    callbacks_list = [earlystop, check_pt]

    trained_model = model.fit(X_train, [Y_train], epochs = EPOCHS, batch_size = BATCH_SIZE, shuffle = True, \
    							validation_data = [X_val, [Y_val]], callbacks = callbacks_list)
            
    return trained_model, model


def model_history(trained_model):

	fig, axs = plt.subplots(1,2,figsize=(15,5))

	axs[0].plot(trained_model.history['loss'])
	axs[0].plot(trained_model.history['val_loss'])
	axs[0].set_title('Model Loss')
	axs[0].set_ylabel('Loss')
	axs[0].set_xlabel('Epoch')
	axs[0].legend(['Train', 'Validation'], loc='upper right')

	axs[1].plot(trained_model.history['acc'])
	axs[1].plot(trained_model.history['val_acc'])
	axs[1].set_title('Model Accuracy')
	axs[1].set_ylabel('Accuracy')
	axs[1].set_xlabel('Epoch')
	axs[1].legend(['Train', 'Validation'], loc='upper right')
	plt.show()


if __name__ == "__main__":
	
	ratio = 6
	data_dir = '../data/WithoutReviews/'
	df = pd.read_hdf(data_dir + 'No_normalized_data.h5')

	x, y = x_value(df, 20, 40, 60, 80)
	print('x and y created')
	x, y = equalSplit(x, y, ratio)
	print('x and y equally splitted')

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
	print('Train and test split of x and y\n')

	# print('Training started')
	# clf = MLPClassifier()
	# clf.fit(x_train, y_train)

	# print('Testing Started')
	# y_pred = clf.predict(x_train)
	# print('Train Accuracy - ', accuracy_score(y_train, y_pred))
	# y_pred = clf.predict(x_test)
	# print('Test Accuracy - ', accuracy_score(y_test, y_pred))
	# print(len(x_train), len(x_test))
	model = load_model(base_dir + 'model_.h5')
	Y_pred = model.predict(X_test)
