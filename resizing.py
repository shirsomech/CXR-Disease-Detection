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

def get_configuration_str(percentage, training_config, target_dataset):
    og_training_datasets = "-".join([str(int((1-percentage)*100)), "_".join([ds.name for ds in training_config])])
    target_db = "-".join([str(int(percentage*100)), target_dataset.__class__.__name__])
    return " ".join([og_training_datasets, target_db])

def create_alexnet(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Modify the last fully connected layer to change the number of output channel
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def main():
    model = create_alexnet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('fine_tuning/PretrainedFourSources.pt'))

    # Baseline
    vietnam_dataset = VinXRay()
    
    train_size = int(0.8 * len(vietnam_dataset))
    test_size = len(vietnam_dataset) - train_size

    # Training 12,000 Testing 3,000
    vietnam_training_dataset, vietnam_testing_dataset = torch.utils.data.random_split(vietnam_dataset, [train_size, test_size])
    
    # Building the datasets for the training configuration matrix
    one_dataset = [
        COVID19_Radiography.Datasets.NORMAL_DS1, 
        COVID19_Radiography.Datasets.LO_DS1
    ]
    two_datasets = one_dataset + [
        COVID19_Radiography.Datasets.NORMAL_DS2,
        COVID19_Radiography.Datasets.VP_DS2
    ]
    four_datasets = two_datasets + [
        COVID19_Radiography.Datasets.COVID_DS3,
        COVID19_Radiography.Datasets.COVID_DS4,
    ]
    six_datasets = four_datasets + [
        COVID19_Radiography.Datasets.COVID_DS5,
        COVID19_Radiography.Datasets.COVID_DS6,
    ]
    eight_datasets = six_datasets + [
        COVID19_Radiography.Datasets.COVID_DS7,
        COVID19_Radiography.Datasets.COVID_DS8,
    ]
    training_dataset_configs = [
        one_dataset, 
        two_datasets, 
        four_datasets, 
        six_datasets, 
        eight_datasets
    ]

    accuracy_matrix = np.empty((len(training_dataset_configs), 2))
    recall_matrix = np.empty((len(training_dataset_configs), 2))
    precision_matrix = np.empty((len(training_dataset_configs), 2))
    f1_score_matrix = np.empty((len(training_dataset_configs), 2))

    for i, training_config in enumerate(training_dataset_configs):
        training_dataset = COVID19_Radiography(training_config)
        for j, value in enumerate([False, True]):
            if value:
                # Resize the training dataset to 12,000
                training_dataset = data_utils.Subset(training_dataset, torch.arange(12000))
                model_path = os.path.join(MODELS_DIR_NAME, f"Resized{len(training_config)}.pt")
            else:
                model_path = os.path.join(MODELS_DIR_NAME, f"NotResized{len(training_config)}.pt")

            datasets = DatasetManager(training_dataset, vietnam_testing_dataset)
            t = TrainAndTest(model, datasets, lr=BASE_LR, output_path=f"{MODELS_DIR_NAME}/results.txt")
            report = t.train(NUM_EPOCHS)

            accuracy_matrix[i][j] = report['accuracy']
            precision_matrix[i][j] = report['macro avg']['precision']
            recall_matrix[i][j] = report['macro avg']['recall']
            f1_score_matrix[i][j] = report['macro avg']['f1-score']

            with open(f"{MODELS_DIR_NAME}/matrices_macro.txt", "a") as f:
                f.write(f"Accuracy: {accuracy_matrix}\n")
                f.write(f"Precision: {precision_matrix}\n")
                f.write(f"Recall: {recall_matrix}\n")
                f.write(f"F1 Score: {f1_score_matrix}\n")
                

if __name__ == "__main__":
  main()