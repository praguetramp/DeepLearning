import keras
from chainer import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from CNN_pre import *

x_train, x_test, y_train, y_test, input_shape = load_and_processing()
num_classes = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=12,
          validation_data=(x_test, y_test)
          )
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss :', score[0])
print('Test accuracy:', score[1])
