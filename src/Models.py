import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.callbacks import *
from scipy.ndimage.filters import *

def unet(input_height, input_width, n_classes):
    inputs = Input(shape=(input_height, input_width, 1))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(strides=(2,2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(strides=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(strides=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format='channels_last')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(strides=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', data_format='channels_last')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Add(name='layer_fusion1')([drop4, up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format='channels_last')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Add(name='layer_fusion2')([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Add(name='layer_fusion3')([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Add(name='layer_fusion4')([conv1, up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_last')(merge9)
    conv9 = Conv2D(64, 4, activation = 'relu', padding = 'same', data_format='channels_last')(conv9)
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', data_format='channels_last')(conv9)

    output = (Activation('softmax'))(conv9)
    return Model(inputs, output)

def fourier_transform(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	img_magnitude_spectrum = 20 * np.log(np.abs(fshift))
	return img_magnitude_spectrum

def laplacian(img):
	laplacian = cv2.Laplacian(img, cv2.CV_64F)
	return laplacian

def canny(img):
	canny = cv2.Canny(img, 100, 200)
	return np.expand_dims(canny, axis=2)

def normalize(img):
	mean = np.mean(img)
	std = np.std(img)
	row = img.shape[0]
	col = img.shape[1]
	for i in range(row):
		for j in range(col):
			img[i][j][0] = (img[i][j][0] - mean) / std
	return img[:, :, :1]

def load_training_data():
        # load the text file with the names of training file names
	train_names = open('./train.txt').read().split() 
	x_train = []
	y_train = []
	n_classes = 3
	
	for file_name in train_names:
		img = cv2.imread('./train_frames/' + file_name + '.png')
		if img.shape[0] != 256 or img.shape[1] != 256:
			img  = cv2.resize(img, (256, 256), \
				interpolation = cv2.INTER_AREA)
		img = median_filter(img, size=3)
		img = normalize(img)
		x_train.append(img)
		img = cv2.imread('./train_masks/' + file_name + '.png')
		if img.shape[0] != 256 or img.shape[1] != 256:
			img  = cv2.resize(img, (256, 256), \
				interpolation = cv2.INTER_AREA)
		img = img[:,:,0]
		seg_mask = np.zeros((img.shape[0], img.shape[1], n_classes))
		for label in range(n_classes):
			seg_mask[:,:,label] = (img == label).astype(int)
		y_train.append(seg_mask)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	return x_train, y_train

def load_testing_data():
        # load the text file of training file names
	test_names = open('./test.txt').read().split()
	x_test = []
	img_shapes = []
	n_classes = 3
	for file_name in test_names:
		img = cv2.imread('./test_frames/' + file_name + '.png')
		img_shapes.append(img.shape)
		if img.shape[0] != 256 or img.shape[1] != 256:
			img  = cv2.resize(img, (256, 256), \
				interpolation = cv2.INTER_AREA)
		img = median_filter(img, size=3)
		img = normalize(img)
		x_test.append(img)
	x_test = np.array(x_test)
	return x_test, img_shapes, test_names
	
#Load in training, test, and img resolution data
x_train, y_train = load_training_data()
x_test, img_shapes, test_names = load_testing_data()

#Select a model, and print out it's architecture summary
model = unet(256, 256, 3)
model.summary()

#Best Optimizer Parameters for U-Net. Reccommend setting lr=0.3 for FCN8
sgd = optimizers.SGD(lr=0.01, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, \
		metrics=['accuracy'])

#Fit model using traning data. Both the best model found during training,
#and the model trained for the entire duration of epochs are saved. For
#FCN8, the recommended batch size is 32.
model_path = './models/Current_Best.h5'
callbacks=[ModelCheckpoint(filepath=model_path, \
		monitor='val_loss', save_best_only=True)]
model.fit(x_train, y_train, batch_size=1, epochs=300, validation_split=0.1, callbacks=callbacks)
model.save('./models/Full.h5')

#Test model trained on the full number of epochs
pred = model.predict(x_test)
pred_imgs = np.argmax(pred, axis=3)
for i in range(len(pred_imgs)):
	output_path = '../predictions/full_outputs/' \
			+ test_names[i] \
			+ '.png'
	cv2.imwrite(output_path, pred_imgs[i])
	if img_shapes[i][0] != 256 or img_shapes[i][1] != 256:
		img = cv2.imread(output_path)
		img = img[:,:,0]
		img = cv2.resize(img, \
			(img_shapes[i][1], img_shapes[i][0]), \
			interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(output_path, img)

#Test best model trained, based on the val_loss
model = keras.models.load_model('./models/Current_Best.h5')
pred = model.predict(x_test)
pred_imgs = np.argmax(pred, axis=3)
for i in range(len(pred_imgs)):
	output_path = './predictions/best_outputs/' \
			+ test_names[i] \
			+ '.png'
	cv2.imwrite(output_path, pred_imgs[i])
	if img_shapes[i][0] != 256 or img_shapes[i][1] != 256:
		img = cv2.imread(output_path)
		img = img[:,:,0]
		img = cv2.resize(img, \
			(img_shapes[i][1], img_shapes[i][0]), \
			interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(output_path, img)
