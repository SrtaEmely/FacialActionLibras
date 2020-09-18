# Import necessary components
import keras
from keras import backend as K
import numpy as np
import os
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection as sk
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LSTM, Reshape
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
'''
##############################################################################
##############################################################################
#CNN
##############################################################################
##############################################################################
'''
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)
##############################################################################
##############################################################################   
x_corpus_upper=np.load("F:/Doutorado/Pesquisa/Python/Data_annotations/x_u_corpus.npy")
x_hm_upper=np.load("F:/Doutorado/Pesquisa/Python/Data_annotations/x_u_HM.npy")
x_disfa_upper=np.load("F:/Doutorado/Pesquisa/Python/cnn_lstm/data/x_u.npy")

y_corpus_upper=np.load("F:/Doutorado/Pesquisa/Python/Data_annotations/y_u_corpus.npy")
y_hm_upper=np.load("F:/Doutorado/Pesquisa/Python/Data_annotations/y_u_HM.npy")
y_disfa_upper=np.load("F:/Doutorado/Pesquisa/Python/cnn_lstm/data/y_u.npy")

x_hm_upper=np.nan_to_num(x_hm_upper)
y_hm_upper=np.nan_to_num(y_hm_upper)
##############################################################################
##############################################################################
X=np.append(np.append(np.append(x_corpus_upper,x_hm_upper),x_disfa_upper))
Y=np.append(np.append(np.append(y_corpus_upper[:int(x_corpus_upper.size/(60*97*1))],y_hm_upper[:int(x_hm_upper.size/(60*97*1))]),y_disfa_upper))
X=np.nan_to_num(X)
Y=np.nan_to_num(Y)
X = X.astype('float32')
##############################################################################
##############################################################################
img_rows, img_cols = 60, 97
X = X.reshape(int(X.shape[0]/(60*97*1)), img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y_train = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(encoded_Y_train)
num_classes = Y.shape[1]
x_train, x_test, y_train, y_test = sk.train_test_split(X,Y,train_size = 0.7, random_state = seed(2018))
print('train shape =',x_train.shape)
print('test shape =', x_test.shape)
##############################################################################
##############################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',fmeasure, precision, recall])

##############################################################################
##############################################################################
batch_size = 128
epochs=15
print('(epocas, batch)=',(epochs,batch_size))

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
  
history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        validation_steps=len(x_test) // batch_size)
##############################################################################
##############################################################################
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
save_dir = os.path.join(os.getcwd(), 'saved_models_corpus+hm+CK+disfa')
y_pred = model.predict(x_test)
y_t=np.argmax(y_test, axis=-1)
y_pred=np.argmax(y_pred, axis=-1)
print(classification_report(y_t, y_pred))
matrix_c = confusion_matrix(y_t, y_pred)
np.save(".../cnn_upper_matrix_c", matrix_c)
##############################################################################
##############################################################################
save_dir = os.path.join(os.getcwd(), 'saved_models_corpus+hm+CK+disfa')
model_name_l = 'cnn_upper_0_60x97.h5'
model.save_weights('E:\Doutorado\Pesquisa\Python\CNN\saved_models_corpus+hm+CK+disfa\cnn_upper_0_60x97.h5py')

model_json = model.to_json()
with open("E:\Doutorado\Pesquisa\Python\CNN\saved_models_corpus+hm+CK+disfa\cnn_upper_0_60x97.json", "w") as json_file:
    json_file.write(model_json)

model_yaml = model.to_yaml()
with open("E:\Doutorado\Pesquisa\Python\CNN\saved_models_corpus+hm+CK+disfa\cnn_upper_0_60x97.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path_l = os.path.join(save_dir, model_name_l)
model.save(model_path_l)
print('Saved trained model at %s ' % model_path_l)
#############################################################################
#############################################################################
# list all data in history
print(history.history.keys())
# summarize history for accuracy
f = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and validation accuracy of the model for the expressions on upper part of the face')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
f.show()
f.savefig('...\CNN_accuracy_upper.png')
f.savefig("...\CNN_accuracy_upper.pdf", bbox_inches='tight')
# summarize history for loss
g = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss of the model for the expressions on upper part of the face')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
g.show()
g.savefig('...\CNN_loss_upper.png')
g.savefig("...\CNN_loss_upper.pdf", bbox_inches='tight')
#plt.close()
