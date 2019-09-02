import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

import knifey
knifey.maybe_download_and_extract()

knifey.copy_files()


model = VGG16(include_top=True, weights='imagenet')

train_dir = knifey.train_dir
test_dir = knifey.test_dir

datagen_train = ImageDataGenerator(
rescale=1./255,
rotation_range=180,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.1,
zoom_range=[0.9, 1.5],
horizontal_flip=True,
vertical_flip=True,
fill_mode='nearest') 
datagen_test = ImageDataGenerator(rescale=1./255)

input_shape = model.layers[0].output_shape[1:3]
batch_size = 20

generator_train = datagen_train.flow_from_directory(directory=train_dir,
							 target_size=input_shape,
							 batch_size=batch_size,
							 shuffle=True)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
						      target_size=input_shape,
						      batch_size=batch_size,
						      shuffle=False)

cls_train = generator_train.classes
num_classes = generator_train.num_classes
steps_test = generator_test.n / batch_size

transfer_layer = model.get_layer('block5_pool')

conv_model = Model(inputs=model.input,
		     outputs=transfer_layer.output)
conv_model.summary()

new_model = Sequential()
new_model.add(conv_model)
new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes, activation='softmax'))
new_model.summary()

for layer in conv_model.layers:
    layer.trainable = False
def print_layer_trainable():
   for layer in conv_model.layers:
	print("{0}:\t{1}".format(layer.trainable, layer.name))
print_layer_trainable()

from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight='balanced',
				       classes=np.unique(cls_train),
					y=cls_train)
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
epochs = 20
steps_per_epoch = 100
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = new_model.fit_generator(generator=generator_train,
				     epochs=epochs,
				     steps_per_epoch=steps_per_epoch,
				     class_weight=class_weight,
				     validation_data=generator_test,
				     validation_steps=steps_test)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.style.use('fivethirtyeight')
plt.xlim(1,20)
plt.ylim(0,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()
plt.xlim(1,20)
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

conv_model.trainable = True
for layer in conv_model.layers:
	trainable = ('block5' in layer.name or 'block4' in layer.name)
	layer.trainable = trainable

optimizer_fine = Adam(lr=1e-7)
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)
history = new_model.fit_generator(generator=generator_train,
					    epochs=epochs,
					    steps_per_epoch=steps_per_epoch,
					    class_weight=class_weight,
					    validation_data=generator_test,
					    validation_steps=steps_test)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.style.use('fivethirtyeight')
plt.xlim(1,20)
plt.ylim(0,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()
plt.xlim(1,20)
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
