import numpy as np

from src.models import lenet, linear_model, nvidia
from src.read import read_images, augment_images

# read CSV file in the provide folder and load all images
# This also includes the side images
images, measurements = read_images('./data_2runs', '\\')

print(np.shape(images))
print(np.shape(measurements))

# Augment images by flipping them horizontally
augmented_images, augmented_measurements = augment_images(images, measurements)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

n_train = np.size(y_train)
image_shape = np.shape(X_train)

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)


# Create the Keras model
model = nvidia()
model.compile(loss='mse', optimizer='adam')

# Callback for saving the model after each check point
from keras.callbacks import ModelCheckpoint
callback = ModelCheckpoint('model-2runs.{epoch:02d}-{val_loss:.5f}.h5')

# Train the model and save the final result
history_object = model.fit(x=X_train, y=y_train, validation_split=0.2, shuffle=True, nb_epoch=10, callbacks=[callback])
model.save('./model-2runs.h5')

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