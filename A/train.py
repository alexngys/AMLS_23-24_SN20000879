 
#  1. Import Packages


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.ndimage import zoom
#pydot and graphviz 


def train_evaluate_taskA():
  pneumonia_dataset = np.load('./Datasets/pneumoniamnist.npz')

  #  3. Get images and labels
  train_images = pneumonia_dataset['train_images']

  val_images = pneumonia_dataset['val_images']

  test_images = pneumonia_dataset['test_images']

  train_labels = pneumonia_dataset['train_labels']

  val_labels = pneumonia_dataset['val_labels']

  test_labels = pneumonia_dataset['test_labels']

  #  4. Display a image from the dataset

  sample_image = train_images[0]
  plt.imshow(sample_image, cmap="gray")
  plt.axis("off")
  plt.title("Sample image from dataset")
  plt.show()

  
  #  5. Convert image data from integers to floating point numbers


  train_images = train_images / 255.0

  val_images = val_images / 255.0

  test_images = test_images / 255.0

  #  20. Generate more images from dataset by applying a zoom filter 

  def generateZoomImages(images_array, labels_array, zoom_levels):
    height, width = np.shape(images_array[0])
    new_images = images_array
    new_labels = labels_array
    for i in zoom_levels:
      zoom_tuple = (i,) * 2 + (1,) * (images_array[0].ndim - 2)
      for j in range(len(labels_array)):
          if labels_array[j][0] == 0:
            zh = int(np.round(height / i))
            zw = int(np.round(width / i))
            top = (height - zh) // 2
            left = (width - zw) // 2
            out = zoom(images_array[j][top:top+zh, left:left+zw], zoom_tuple)
            trim_top = ((out.shape[0] - height) // 2)
            trim_left = ((out.shape[1] - width) // 2)
            out = out[trim_top:trim_top+height, trim_left:trim_left+width]
            new_images = np.append(new_images, [out], axis=0)  
            new_labels = np.append(new_labels,[labels_array[j]], axis=0)
          
    return new_images, new_labels

  print("\nLoading images \n")
  new_train_images, new_train_labels = generateZoomImages(train_images, train_labels, [1.1, 1.2])

  #  21. Train NN with new training dataset

  NN_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu') ,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2 , activation ='softmax')
  ])

  NN_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

  NN_model.compile(optimizer=NN_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  print("Training NN")

  history = NN_model.fit(new_train_images, new_train_labels, validation_data=(val_images, val_labels), batch_size=50, epochs=40)


  # Loss plot
  figure, axis = plt.subplots(2)
  line1, = axis[0].plot(history.history['loss'], 'r')
  line2, = axis[0].plot(history.history['val_loss'], 'b')
  axis[0].legend(['Train','Validation'])
  axis[0].set_title("Loss")

  # Accuracy plot
  line3, = axis[1].plot(history.history['accuracy'], 'r')
  line4, = axis[1].plot(history.history['val_accuracy'], 'b')
  axis[1].legend(['Train','Validation'])
  axis[1].set_title("Accuracy")
  figure.suptitle("NN training and evaluation results")
  figure.tight_layout(pad=1)

  plt.show()

  
  #  22. Evaluate NN model with test dataset


  NN_test_prediction = NN_model.predict(test_images)

  NN_test_prediction = np.argmax(NN_test_prediction, axis=1)

  total = 0
  correct = 0
  for i in range(len(NN_test_prediction)):
    if NN_test_prediction[i] == test_labels[i][0]:
      correct += 1
    total += 1

  print("Accuracy:", correct * 100 / total)

  result = confusion_matrix(test_labels, NN_test_prediction , normalize='true')

  cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = ['Negative','Positive'])

  cm_display.plot()
  plt.title("Confusion matrix of test dataset NN prediction", pad=10)
  plt.show()


  
  #  24. Train CNN with the new training dataset


  CNN_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size =(5, 5), strides =1, activation ='relu', input_shape = (28,28,1)), 
    tf.keras.layers.MaxPooling2D(pool_size =(3, 3), strides =(1, 1)),
    tf.keras.layers.Conv2D(32, kernel_size =(5, 5), strides =1, activation ='relu'), 
    tf.keras.layers.MaxPooling2D(pool_size =(3, 3), strides =(1, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu') ,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2 , activation ='softmax')
  ])

  optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
  CNN_model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

  print("Training CNN")
  CNN_history = CNN_model.fit(new_train_images, new_train_labels, validation_data=(val_images, val_labels), batch_size=50, epochs=25)


  # Loss plot
  figure, axis = plt.subplots(2)
  line1, = axis[0].plot(CNN_history.history['loss'], 'r')
  line2, = axis[0].plot(CNN_history.history['val_loss'], 'b')
  axis[0].legend(['Train','Validation'])
  axis[0].set_title("Loss")

  # Accuracy plot
  line3, = axis[1].plot(CNN_history.history['accuracy'], 'r')
  line4, = axis[1].plot(CNN_history.history['val_accuracy'], 'b')
  axis[1].legend(['Train','Validation'])
  axis[1].set_title("Accuracy")
  figure.suptitle("CNN training and evaluation results")
  figure.tight_layout(pad=1)

  plt.show()

  
  #  25. Evaluate CNN model with test dataset


  CNN_test_prediction = CNN_model.predict(test_images)

  CNN_test_prediction = np.argmax(CNN_test_prediction, axis=1)

  total = 0
  correct = 0
  for i in range(len(CNN_test_prediction)):
    if CNN_test_prediction[i] == test_labels[i][0]:
      correct += 1
    total += 1

  print("Accuracy:", correct * 100 / total)

  result = confusion_matrix(test_labels, CNN_test_prediction , normalize='true')

  cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = ['Negative','Positive'])

  cm_display.plot()
  plt.title("Confusion matrix of test dataset CNN prediction", pad=10)
  plt.show()




