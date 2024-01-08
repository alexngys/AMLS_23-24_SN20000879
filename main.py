import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.ndimage import zoom
from A.train import train_evaluate_taskA
from B.train import train_evaluate_taskB
 
def main():
  
  while True:
    task_choice = input("Enter A or B to select the task: ")
    if task_choice == "A" or task_choice == "B" or task_choice == "close":
      break
    else:
      print("Invalid input")
    
  if task_choice == "A":
    print("\nRunning modified training and testing program for task A from train.py")
    print("\nIf you would like to see how the models were fully trained in greater detail please refer to the train.ipynb notebook in the task folder.")
    train_evaluate_taskA()
    
  

  elif task_choice == "B":
    print("\nRunning modified training and testing program for task A from train.py")
    print("\nIf you would like to see how the models were fully trained in greater detail please refer to the train.ipynb notebook in the task folder.\n")
    train_evaluate_taskB()

  else:
    pass

if __name__ == "__main__":
  main()