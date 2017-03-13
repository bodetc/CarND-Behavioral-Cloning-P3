import numpy as np

from src.models import lenet, linear_model, nvidia
from src.read import read_images, augment_images

images, measurements = read_images('./data', '\\')

# Augment images by flipping them horizontally
augmented_images, augmented_measurements = augment_images(images, measurements)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

n_train = np.size(y_train)
image_shape = np.shape(X_train)

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)

model = nvidia()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(x=X_train, y=y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

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