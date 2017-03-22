import numpy as np

from src.generator import read_csv, generator
from src.models import nvidia

# Read the csv files into samples
samples = read_csv('./data', '\\')
samples.extend(read_csv('./data_sim', '/'))

print("Number of lines in csv =", np.alen(samples))

# Split into training and validation dataset
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

n_train = np.alen(train_samples) * 6
n_validation = np.alen(validation_samples) * 6
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)

# Create the generator functions
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create the Keras model
model = nvidia()
model.compile(loss='mse', optimizer='adam')

# Callback for saving the model after each check point
from keras.callbacks import ModelCheckpoint
callback = ModelCheckpoint('model.{epoch:02d}-{val_loss:.5f}.h5')

# Train the model using the generator function and save the final result
history_object = \
    model.fit_generator(train_generator, samples_per_epoch=n_train,
                        validation_data=validation_generator, nb_val_samples=n_validation,
                        nb_epoch=10, callbacks=[callback])
model.save('./model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
