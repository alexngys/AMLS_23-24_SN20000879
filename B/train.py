
#  1. Import Packages


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.ndimage import zoom
# pydot and graphviz 


#  2. Import datasets

def train_evaluate_taskB():
  path_dataset = np.load('./Datasets/pathmnist.npz')



  #  3. Get images and labels


  train_images = path_dataset['train_images']

  val_images = path_dataset['val_images']

  test_images = path_dataset['test_images']

  train_labels = path_dataset['train_labels']

  val_labels = path_dataset['val_labels']

  test_labels = path_dataset['test_labels']


  def getClassDistribution(label_array, title): 
    class_count = [0,0,0,0,0,0,0,0,0]
    for i in label_array:
        class_count[i[0]] += 1
    def my_fmt(x):
      print(x)
      return '{:.1f}%\n({:.0f})'.format(x, total*x/100)
    total =  sum(class_count)
    fig = plt.figure()
    plt.pie(class_count, labels=['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle','normal colon mucosa', 'cancer-associated stroma', 'colorectal carcinoma'], autopct=my_fmt) # colorectal adenocarcinoma epithelium
    plt.title(title)


  getClassDistribution(train_labels, "Distribution of training dataset")
  getClassDistribution(val_labels, "Distribution of evaluation dataset")
  getClassDistribution(test_labels, "Distribution of testing dataset")


  #  4. Display images of each class from the dataset


  indexes = [0,0,0,0,0,0,0,0,0]
  i = 0
  while 0 in indexes:
    if indexes[train_labels[i][0]] == 0:
      indexes[train_labels[i][0]] = i
    i += 1

  classes = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle','normal colon mucosa', 'cancer-associated stroma', 'colorectal carcinoma']

  for i in range(len(classes)):
    sample_image = train_images[indexes[i]]
    plt.imshow(sample_image, cmap="gray")
    plt.axis("off")
    plt.title(f"Sample image from dataset of {classes[i]}")
    plt.show()


  #  5. Convert image data from integers to floating point numbers


  train_images = train_images / 255.0

  val_images = val_images / 255.0

  test_images = test_images / 255.0


  #  6. Create a neural network (NN) model


  NN_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu') ,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(9 , activation ='softmax')
  ])

  NN_model.summary()

  tf.keras.utils.plot_model(NN_model, to_file='NN_model.png', show_shapes=True, 
      show_dtype=True)


  #  7. Compile NN model


  NN_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

  NN_model.compile(optimizer=NN_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


  #  8. Train NN Model

  print("Training NN")

  history = NN_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=100, epochs=40)


  #  9. Plot NN training and evaluation results


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


  #  10. Run NN model with test data and plot confusion matrix


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

  labels = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle','normal colon mucosa', 'cancer-associated stroma', 'colorectal carcinoma']
  cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = labels)

  cm_display.plot(values_format = '.2f')
  plt.title("Confusion matrix of test dataset NN prediction", pad=10)
  plt.xticks(rotation=45, ha='right')
  plt.show()





  #  12. Create a convolutional neural network (CNN) model


  CNN_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size =(3, 3), strides =1, activation ='relu', input_shape = (28,28,3)), 
    tf.keras.layers.MaxPooling2D(pool_size =(3, 3), strides =(1, 1)),
    tf.keras.layers.Conv2D(32, kernel_size =(3, 3), strides =1, activation ='relu'), 
    tf.keras.layers.MaxPooling2D(pool_size =(3, 3), strides =(1, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu') ,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(9 , activation ='softmax')
  ])


  CNN_model.summary()

  tf.keras.utils.plot_model(CNN_model, to_file='CNN_model.png', show_shapes=True, 
      show_dtype=True)


  #  13. Complie CNN model


  optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
  CNN_model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


  #  14. Train CNN model

  print("Training CNN")

  CNN_history = CNN_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=50, epochs=40)


  #  15. Plot CNN training and evaluation results


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


  #  16. Run CNN model with test data and plot confusion matrix


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

  labels = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle','normal colon mucosa', 'cancer-associated stroma', 'colorectal carcinoma']
  cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = labels)

  cm_display.plot(values_format = '.2f')
  plt.title("Confusion matrix of test dataset CNN prediction", pad=10)
  plt.xticks(rotation=45, ha='right')
  plt.show()





