import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

ds_path = sys.argv[1]
if not os.path.isdir(ds_path):
    raise ValueError("The informed path does not exist. Please try to run again with a valid folder path.")

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
                    img = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
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

resnet_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

x=resnet_model.output

x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x) 
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dense(512,activation='relu')(x)
preds=tf.keras.layers.Dense(num_classes,activation='softmax')(x)

#criando o objeto do modelo
model=tf.keras.models.Model(inputs=resnet_model.input,outputs=preds)

#exibindo a arquitetura final
model.summary()

#Defines the optimizer, loss function and metrics
model.compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#Trains the model to classify the images on the training set
model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
model.save('resnet50_model_weights.h5')
json_model_arc = model.to_json()
with open('resenet50_model_arc.json','w+') as f:
   f.write(json_model_arc)
