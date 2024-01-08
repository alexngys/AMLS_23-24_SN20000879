import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_prediction(prediction, labels):
  print("\nIf you would like to see how the models where trained please refer to the train.ipynb notebook in the task folder.")
  prediction = np.argmax(prediction, axis=1)

  total = 0
  correct = 0
  for i in range(len(prediction)):
    if prediction[i] == labels[i][0]:
      correct += 1
    total += 1

  print("\nAccuracy:", correct * 100 / total)

  result = confusion_matrix(labels, prediction , normalize='true')

  cm_display = ConfusionMatrixDisplay(confusion_matrix = result, display_labels = ['Negative','Positive'])

  cm_display.plot()
  plt.title("Confusion matrix of test dataset model prediction", pad=10)
  plt.show()

def main():
  
  while True:
    task_choice = input("Enter A or B to select the task: ")
    if task_choice == "A" or task_choice == "B" or task_choice == "close":
      break
    else:
      print("Invalid input")
    
  if task_choice == "A":
    print("\nLoading dataset for task A...")
    pneumonia_dataset = np.load('Datasets/pneumoniamnist.npz')
    test_images = pneumonia_dataset['test_images']
    test_images = test_images / 255.0
    test_labels = pneumonia_dataset['test_labels']
    print("\nLoaded dataset")

    while True:
      model_choice = input("\nSelect NN or CNN model would you like to evaluate: ")
      if model_choice == "NN" or model_choice == "CNN" or model_choice == "close":
        break
      else:
        print("Invalid input")
    
    if model_choice == "NN":
      print("\nRunning evaluation of NN for task A...")
      model = tf.keras.models.load_model('A/NN_model')
      test_prediction = model.predict(test_images)
      evaluate_prediction(test_prediction, test_labels)

    elif model_choice == "CNN":
      print("\nRunning evaluation of CNN for task A...")
      model = tf.keras.models.load_model('A/CNN_model')
      test_prediction = model.predict(test_images)
      evaluate_prediction(test_prediction, test_labels)

    else: 
      pass
  

  elif task_choice == "B":
    print("\nLoading dataset for task B...")
    path_dataset = np.load('Datasets/pathmnist.npz')
    test_images = path_dataset['test_images']
    test_images = test_images / 255.0
    test_labels = path_dataset['test_labels']
    print("\nLoaded dataset")

    while True:
      model_choice = input("\nSelect NN or CNN model would you like to evaluate: ")
      if model_choice == "NN" or model_choice == "CNN" or model_choice == "close":
        break
      else:
        print("Invalid input")
    
    if model_choice == "NN":
      print("\nRunning evaluation of NN for task B...")
      model = tf.keras.models.load_model('B/NN_model')
      test_prediction = model.predict(test_images)
      evaluate_prediction(test_prediction, test_labels)

    elif model_choice == "CNN":
      print("\nRunning evaluation of CNN for task B...")
      model = tf.keras.models.load_model('B/CNN_model')
      test_prediction = model.predict(test_images)
      evaluate_prediction(test_prediction, test_labels)

    else: 
      pass

  else:
    pass

if __name__ == "__main__":
  main()