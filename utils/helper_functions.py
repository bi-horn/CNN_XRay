# -*- coding: utf-8 -*-
"""
Helper functions

"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.inception import InceptionOutputs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_auc_score, roc_curve, auc

import os, time, random, torch, warnings
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score


#Data Preprocessing
def data_preprocess(data_path, sample_ratio, batch_size):
  # Create data transforms
  data_transforms = transforms.Compose([
    #standard measures if you want to use e.g. ResNet18
    transforms.Resize((224, 224)), #Consistent Formatting: Ensure uniform size and tensor format for inputs
    transforms.RandomHorizontalFlip(), #Data Augmentation: Enhance dataset diversity with random transformations
    transforms.ToTensor(), #datset to 4D tensor (# of images, height, width, channels)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Standardization: Normalize pixel values for consistent convergence

  # Get dataset from folder and apply data transforms
  dataset = datasets.ImageFolder(root = "{}data".format(data_path), transform = data_transforms)


  # Get a sample of the data randomly
  num_samples = int(len(dataset) * sample_ratio)
  indices = np.random.choice(range(len(dataset)), num_samples, replace = False)

  # Split the data into training, test, and validation sets
  train_size = int(0.7 * num_samples) #70% of the dataset is for training
  test_size = int(0.2 * num_samples) #20% of the dataset is for testing
  val_size = num_samples - train_size - test_size #10% of the dataset is for validation

  train_indices = indices[ : train_size]
  test_indices = indices[train_size : train_size + test_size]
  val_indices = indices[train_size + test_size : ]

  #Load the dataset into a format that PyTorch can use, such as a torch.utils.data.Dataset.
  # Create random training, test, and validation datasets

  samples = [torch.utils.data.sampler.SubsetRandomSampler(i) for i in [train_indices, test_indices, val_indices]]

  train_loader = DataLoader(dataset, batch_size = batch_size, sampler = samples[0], num_workers = 4, pin_memory = True)
  test_loader = DataLoader(dataset, batch_size = batch_size, sampler = samples[1], num_workers = 4, pin_memory = True)
  val_loader = DataLoader(dataset, batch_size = batch_size, sampler = samples[2], num_workers = 4, pin_memory = True)

  return dataset, train_loader, train_indices, test_loader, test_indices, val_loader, val_indices

#save the metrics after training
def save_metrics(loss, accuracy, validation_loss, validation_accuracy, model, data_path):
    np.save("{}{}_train_loss.npy".format(data_path, model), loss)
    np.save("{}{}_train_accuracy.npy".format(data_path, model), accuracy)
    np.save("{}{}_validation_loss.npy".format(data_path, model), validation_loss)
    np.save("{}{}_validation_accuracy.npy".format(data_path, model), validation_accuracy)

def train_model(model, data_path, device, criterion, optimizer, model_name, num_epochs, train_loader, train_indices, test_loader, test_indices, val_loader, val_indices):

  start1_time = time.time()

  #Create tracking variables
  losses = []
  accuracies = []
  true = []
  pred = []
  v_accuracies = []
  v_losses = []


  #for loop through epochs
  for epoch in range(num_epochs):
    train_loss = 0
    train_accuracy = 0
    start_time = time.time() #set the timer

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar: #Progress Bar Initialization
      #Training Loop
      for X_train, y_train in train_loader: #Iterates through the training dataset using the train_loader; each iteration processes a batch of input data and their corresponding labels
        #Moves the input data and labels to the computing device (e.g., GPU)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        outputs = model(X_train)
        #Prediction and Loss Calculation
        #Add up correct predictions
        _, preds = torch.max(outputs.logits if isinstance(outputs, InceptionOutputs) else outputs, dim = 1) #logits = raw scores produced by the final layer of the Inception model before applying a softmax activation function
        #Calculates the loss between the model predictions and the actual labels using the specified loss criterion
        loss = criterion(outputs.logits if isinstance(outputs, InceptionOutputs) else outputs, y_train)

        #Backpropagation and Optimization
        optimizer.zero_grad() #Resets the gradients of the optimizer to zero
        loss.backward()
        optimizer.step()
        #Metrics Calculation
        train_loss = loss.item() * X_train.size(0) #Computes the loss for the current batch (train_loss) and accumulates it over all batches
        #how many correct from that batch
        train_accuracy += torch.sum(preds == y_train.data)
        #Data Aggregation
        pred.extend(preds.cpu().numpy()) #Extends lists (pred and true) with the predictions and true labels for the current batch, respectively
        true.extend(y_train.cpu().numpy())

        #print results
        pbar.set_postfix({'Accuracy': train_accuracy.item()/len(train_indices), 'Loss': train_loss/len(train_indices), 'Precision': precision_score(true, pred, average='macro'), 'Recall': recall_score(true, pred, average='macro'), 'F1 Score': f1_score(true, pred, average = 'macro')})
        pbar.update()

    #evaluate model for validation dataset
    val_accuracy, val_loss, val_true, val_pred = evaluate_model(model, val_loader, val_indices, 'VALIDATION', criterion, data_path, model_name)

    v_accuracies.append(val_accuracy)
    v_losses.append(val_loss)
    losses.append(train_loss/len(train_indices))
    accuracies.append(train_accuracy.item()/len(train_indices))

  save_metrics(losses, accuracies, v_losses, v_accuracies, model_name, data_path)

  current_time = time.time()
  total = current_time - start1_time
  print(f'Training took: {total/60} minutes')

  return losses, accuracies, v_accuracies, v_losses

def evaluate_model(model, device, dataloader, data_size, dtype, criterion, data_path, model_name):
  _loss, _pred, _true, _accuracy = 0.0, [], [], []
  model.eval()

  with torch.no_grad():
    for X_train, y_train in dataloader:
      X_train = X_train.to(device)
      y_train = y_train.to(device)

      outputs = model(X_train)
      loss = criterion(outputs, y_train)

      _loss += loss.item() * X_train.size(0) #Accumulate the loss for the current sample
      _, predicted = torch.max(outputs.data, 1) #Compute the predicted class indices for each element in the sample: find the indices of the maximum values along the second dimension of the output tensors
      _pred.extend(predicted.cpu().numpy()) #Append the predicted class indices to the _pred list
      _true.extend(y_train.cpu().numpy()) #Append the true class indices to the _true list

  _loss /= len(data_size) #Calculate the average loss by dividing the total loss by the size of the dataset
  _accuracy = accuracy_score(_true, _pred) #Compute the accuracy score by comparing the true labels with the predicted labels
  #The average='macro' parameter calculates the score for each class independently and then averages them
  _recall = recall_score(_true, _pred, average='macro') #Measures the ability of the classifier to identify all relevant instances
  _precision = precision_score(_true, _pred, average='macro') #Measures the ability of the classifier not to flag a negative sample as positive
  _fscore = f1_score(_true, _pred, average='macro') # F1 score = harmonic mean of precision and recall

  print('{}: Accuracy: {:.4f} | Loss: {:.4f} | Recall: {:.4f} | Precision: {:.4f} | F-score: {:.4f}'.format(dtype, _accuracy, _loss, _recall, _precision, _fscore))
  print("")

  return _accuracy, _loss, _true, _pred



def plot_model_curves(losses, accuracies, v_accuracies, v_losses, data_path, model_name):
  #Plotting the Loss and Accuracy Curves

  # Set global font size for labels
  plt.rc('xtick', labelsize=12)    # Set x-axis label size to 12
  plt.rc('ytick', labelsize=12)    # Set y-axis label size to 12
  plt.rc('axes', labelsize=14)     # Set axes label size to 14
  plt.rc('figure', titlesize=16)     # Set title size to 16

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  fig.suptitle('Training and Validation Loss Curve - {}'.format(model_name))

  y_step = 0.1
  x_step = 2

  ax1.plot(losses, label = "Training Loss", color='darkblue')
  ax1.plot(v_losses, label = "Validation Loss", color='lightblue')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  # Set the y-axis limits for the first subplot
  ax1.set_ylim(-0.1, 1.1)
  ax1.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax1.set_xticks(range(0, 20, x_step))
  ax1.legend()

  ax2.plot(accuracies, label = "Training Accuracy", color='darkblue')
  ax2.plot(v_accuracies, label = "Validation Accuracy", color='lightblue')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim(-0.1, 1.1)
  ax2.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax2.set_xticks(range(0, 20, x_step))
  ax2.legend(loc='lower right')


  # Moving the legend outside the plot
  #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


  plt.savefig("{}{}_loss_accuracy.png".format(data_path, model_name))  # Save the figure

  plt.show()

#Evaluate Model on Test Set
def plot_confusion_mat(dataset, _true, _pred, model_name, dataloader, dtype, data_path):
    # calculate confusion matrix
    cm = confusion_matrix(_true, _pred)

    plt.figure(figsize=(8, 8))

    #Create ConfusionMatrixDisplay object with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix for {} dataset - {}".format(dtype, model_name))

    plt.savefig("{}{}_{}_confusion_mat.png".format(data_path, model_name, dtype))  # Save the figure

    plt.show()

# Generates ROC plot and returns AUC using sklearn

#y_true: true binary values
# y_score: Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
def generate_roc(y_true, y_score, data_path, model_name, pos_label = 1): #if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1
  #false positive rate (FPR) and true positive rate (TPR) for different threshold values.
  fpr, tpr, _ = roc_curve(y_true, y_score, pos_label = pos_label)
  roc_auc = auc(fpr, tpr)
  print("fpr:", fpr)
  print("tpr:", tpr)
  print("roc_auc:", roc_auc)
  plt.figure()
  plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], "k--")
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver operating characteristic curve")
  plt.savefig("{}{}_ROC.png".format(data_path, model_name))  # Save the figure
  plt.show()

  print(roc_auc)

def plot_model_curves_comparison(losses, accuracies, v_accuracies, v_losses, losses_np, accuracies_np, v_accuracies_np, v_losses_np, data_path, model_name):
  #Plotting the Loss and Accuracy Curves

  # Set global font size for labels
  plt.rc('xtick', labelsize=12)    # Set x-axis label size to 12
  plt.rc('ytick', labelsize=12)    # Set y-axis label size to 12
  plt.rc('axes', labelsize=14)     # Set axes label size to 14
  plt.rc('figure', titlesize=16)     # Set title size to 16

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

  fig.suptitle('Training and Validation Loss Curve Comparison\n between pretrained and not pretrained {} model'.format(model_name))

  y_step = 0.1
  x_step = 2

  ax1.plot(losses, label = "Training Loss - pretrained", color='darkblue')
  ax1.plot(v_losses, label = "Validation Loss - pretrained", color='lightblue')
  #ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  # Set the y-axis limits for the first subplot
  ax1.set_ylim(-0.1, 1.1)
  ax1.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax1.set_xticks(range(0, 20, x_step))
  ax1.legend()

  ax2.plot(accuracies, label = "Training Accuracy - pretrained", color='darkblue')
  ax2.plot(v_accuracies, label = "Validation Accuracy - pretrained", color='lightblue')
  #ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim(-0.1, 1.1)
  ax2.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax2.set_xticks(range(0, 20, x_step))
  ax2.legend(loc='lower right')

  ax3.plot(losses_np, label = "Training Loss", color='red')
  ax3.plot(v_losses_np, label = "Validation Loss", color='orange')
  ax3.set_xlabel('Epoch')
  ax3.set_ylabel('Loss')
  # Set the y-axis limits for the first subplot
  ax3.set_ylim(-0.1, 1.1)
  ax3.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax3.set_xticks(range(0, 20, x_step))
  ax3.legend()

  ax4.plot(accuracies_np, label = "Training Accuracy", color='red')
  ax4.plot(v_accuracies_np, label = "Validation Accuracy", color='orange')
  ax4.set_xlabel('Epoch')
  ax4.set_ylabel('Accuracy')
  ax4.set_ylim(-0.1, 1.1)
  ax4.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax4.set_xticks(range(0, 20, x_step))
  ax4.legend(loc='lower right')


  # Moving the legend outside the plot
  #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


  plt.savefig("{}{}_loss_accuracy_comparison.png".format(data_path, model_name))  # Save the figure

  plt.show()

def plot_model_curves_comp_with_simple(losses, accuracies, v_accuracies, v_losses, losses_sp, accuracies_sp, v_accuracies_sp, v_losses_sp, data_path):
  #Plotting the Loss and Accuracy Curves

  # Set global font size for labels
  plt.rc('xtick', labelsize=12)    # Set x-axis label size to 12
  plt.rc('ytick', labelsize=12)    # Set y-axis label size to 12
  plt.rc('axes', labelsize=14)     # Set axes label size to 14
  plt.rc('figure', titlesize=16)     # Set title size to 16

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

  fig.suptitle('Training and Validation Loss Curve Comparison\n between ResNet34 and Simple CNN')

  y_step = 0.1
  x_step = 2

  ax1.plot(losses, label = "Training Loss - ResNet34", color='darkblue')
  ax1.plot(v_losses, label = "Validation Loss - ResNet34", color='lightblue')
  #ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  # Set the y-axis limits for the first subplot
  ax1.set_ylim(-0.1, 1.1)
  ax1.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax1.set_xticks(range(0, 20, x_step))
  ax1.legend()

  ax2.plot(accuracies, label = "Training Accuracy - ResNet34", color='darkblue')
  ax2.plot(v_accuracies, label = "Validation Accuracy - ResNet34", color='lightblue')
  #ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim(-0.1, 1.1)
  ax2.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax2.set_xticks(range(0, 20, x_step))
  ax2.legend(loc='lower right')

  ax3.plot(losses_sp, label = "Training Loss - Simple Model", color='red')
  ax3.plot(v_losses_sp, label = "Validation Loss - Simple Model", color='orange')
  ax3.set_xlabel('Epoch')
  ax3.set_ylabel('Loss')
  # Set the y-axis limits for the first subplot
  ax3.set_ylim(-0.1, 1.1)
  ax3.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax3.set_xticks(range(0, 20, x_step))
  ax3.legend()

  ax4.plot(accuracies_sp, label = "Training Accuracy - Simple Model", color='red')
  ax4.plot(v_accuracies_sp, label = "Validation Accuracy - Simple Model", color='orange')
  ax4.set_xlabel('Epoch')
  ax4.set_ylabel('Accuracy')
  ax4.set_ylim(-0.1, 1.1)
  ax4.set_yticks([i * y_step for i in range(int(1 / y_step) + 1)])
  ax4.set_xticks(range(0, 20, x_step))
  ax4.legend(loc='lower right')


  # Moving the legend outside the plot
  #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


  plt.savefig("{}_loss_accuracy_comparison_ResNet34_vs_Simple.png".format(data_path))  # Save the figure

  plt.show()
