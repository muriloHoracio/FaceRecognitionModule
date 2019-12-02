import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def load_data(ds_path, img_size=64):
    classes = [_dir for _dir in os.listdir(ds_path+'/Train') if os.path.isdir(ds_path+'/Train/'+_dir)]
    fail_counter = 0
    x_train, y_train, x_test, y_test = [], [], [], []
    for step in ['Train','Test']:
        for c in classes:
            y = classes.index(c)
            path = os.path.join(ds_path,step,c)
            for img in os.listdir(path):
                try:
                    img = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size,img_size))
                    if step == 'Train':
                        y_train.append(y)
                        x_train.append(img)
                    else:
                        y_test.append(y)
                        x_test.append(img)
                except Exception as e:
                    fail_counter += 1
            
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    shuffled_indeces = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    x_train = x_train[shuffled_indeces]
    y_train = y_train[shuffled_indeces]
    
    shuffled_indeces = np.random.choice(x_test.shape[0], x_test.shape[0], replace=False)
    x_test = x_test[shuffled_indeces]
    y_test = y_test[shuffled_indeces]
    return (x_train, y_train), (x_test, y_test), len(classes)

(x_train, y_train), (x_test, y_test), num_classes = load_data('faces')
print('Train images shape: '+str(x_train.shape))
print('Train labels shape: '+str(y_train.shape))
print('Test images shape: '+str(x_test.shape))
print('Test labels shape: '+str(y_test.shape))

#Normalize all pixels to the interval between 0 and 1
x_train, x_test = x_train/255.0, x_test/255.0
#Adds a channel axis to the images
x_train, x_test = x_train[:,:,:,np.newaxis], x_test[:,:,:,np.newaxis]

with tf.device('/gpu:0'):
    #Defines the sequential model to classify the faces
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(5,5),padding="same",activation="relu",input_shape=(64,64,1)),
        tf.keras.layers.MaxPool2D((2,2),(2,2)),
        tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),
        tf.keras.layers.MaxPool2D((2,2),(2,2)),
        tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),
        tf.keras.layers.MaxPool2D((2,2),(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
  
    #Shows the layers, output shapes and number of parameters
    model.summary()

    #Defines the optimizer, loss function and metrics
    model.compile(
        optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    #Trains the model to classify the images on the training set
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))