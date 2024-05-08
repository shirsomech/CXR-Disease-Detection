import torch
import torch.utils.data as data_utils
import torch.nn as nn
from torchvision import models
from CXRDataset import COVID19_Radiography, VinXRay
from config import *
from TrainAndTest import TrainAndTest
from DatasetManager import DatasetManager
import numpy as np
import os

def create_alexnet(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Modify the last fully connected layer to change the number of output channel
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def main():

    # Baseline
    vietnam_dataset = VinXRay()
    
    train_size = int(0.8 * len(vietnam_dataset))
    test_size = len(vietnam_dataset) - train_size

    # Training 12,000 Testing 3,000
    vietnam_training_dataset, vietnam_testing_dataset = torch.utils.data.random_split(vietnam_dataset, [train_size, test_size])
    
    #datasets = DatasetManager(vietnam_training_dataset, vietnam_testing_dataset)
    #t = TrainAndTest(model, datasets, lr=BASE_LR)

    # Baseline model 
    #report = t.train(NUM_EPOCHS)

    #with open(f"{MODELS_DIR_NAME}/matrices.txt", "a") as f:
    #    f.write(str(report))
    
    #t.save("Baseline", os.path.join(MODELS_DIR_NAME, "baseline.pt"))
    
    # Building the datasets for the training configuration matrix

    AMOUNTS = [500, 1000, 1500, 2000]

    accuracy_matrix = np.empty((4, 1))
    recall_matrix = np.empty((4, 1))
    precision_matrix = np.empty((4, 1))
    f1_score_matrix = np.empty((4, 1))

    model = create_alexnet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("fine_tuning/PretrainedFourSources.pt"))

    for j, amount in enumerate(AMOUNTS):
        
        training_set = data_utils.Subset(vietnam_training_dataset, torch.arange(amount))
        datasets = DatasetManager(training_set, vietnam_testing_dataset)
        
        t = TrainAndTest(model, datasets, lr=BASE_LR, output_path=f"{MODELS_DIR_NAME}/results.txt")
        report = t.train(NUM_EPOCHS)

        t.save("{amount}:", f"{MODELS_DIR_NAME}results.pt")

        accuracy_matrix[j] = report['accuracy']
        precision_matrix[j] = report['macro avg']['precision']
        recall_matrix[j] = report['macro avg']['recall']
        f1_score_matrix[j] = report['macro avg']['f1-score']

        with open(f"{MODELS_DIR_NAME}/matrices_macro.txt", "a") as f:
            f.write(f"Accuracy: {accuracy_matrix}\n")
            f.write(f"Precision: {precision_matrix}\n")
            f.write(f"Recall: {recall_matrix}\n")
            f.write(f"F1 Score: {f1_score_matrix}\n")

        accuracy_matrix[j] = report['accuracy']
        precision_matrix[j] = report['weighted avg']['precision']
        recall_matrix[j] = report['weighted avg']['recall']
        f1_score_matrix[j] = report['weighted avg']['f1-score']
        
        with open(f"{MODELS_DIR_NAME}/matrices_weights.txt", "a") as f:
            f.write(f"Accuracy: {accuracy_matrix}\n")
            f.write(f"Precision: {precision_matrix}\n")
            f.write(f"Recall: {recall_matrix}\n")
            f.write(f"F1 Score: {f1_score_matrix}\n")
            

if __name__ == "__main__":
  main()