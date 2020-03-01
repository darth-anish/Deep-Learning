
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


classifier=Sequential()

classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'sigmoid'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True, rotation_range=40, brightness_range=[0.2,1.5]
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

face_model=classifier.fit_generator(training_set,
                         steps_per_epoch =len(training_set) ,
                         nb_epoch = 20,
                         validation_data = test_set,
                         validation_steps=len(test_set) )

classifier.save('model-cnn15.h5')
print(classifier.summary())

print(training_set.class_indices)

plt.plot(face_model.history['acc'], label='train')
plt.plot(face_model.history['val_acc'], label='test')
plt.axis([0,21,0,1])
plt.title('Model accuracy-CNN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(face_model.history['loss'], label='train')
plt.plot(face_model.history['val_loss'], label='test')
plt.axis([0,21,0,1])
plt.title('Model Loss-CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

