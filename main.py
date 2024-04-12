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

def create_resnet(in_channels, num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def create_alexnet(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Modify the last fully connected layer to change the number of output channel
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def main():
    model = create_alexnet(num_classes=NUM_CLASSES)

    # Baseline
    vietnam_dataset = VinXRay()
    
    train_size = int(0.8 * len(vietnam_dataset))
    test_size = len(vietnam_dataset) - train_size
    vietnam_training_dataset, vietnam_testing_dataset = torch.utils.data.random_split(vietnam_dataset, [train_size, test_size])
    
    datasets = DatasetManager(vietnam_training_dataset, vietnam_testing_dataset)
    t = TrainAndTest(model, datasets, lr=BASE_LR)

    # Baseline model 
    report = t.train(NUM_EPOCHS)

    with open("models/matrices.txt", "a") as f:
        f.write(str(report))
    
    t.save("Baseline", os.path.join(MODELS_DIR_NAME, "baseline.pt"))
    
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

    accuracy_matrix = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    recall_matrix = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    precision_matrix = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    f1_score_matrix = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))

    accuracy_matrix1 = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    recall_matrix1 = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    precision_matrix1 = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))
    f1_score_matrix1 = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)))


    for i, training_config in enumerate(training_dataset_configs):
        for j, percentage in enumerate(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET):

            print(f"Training on %{(1-percentage)*100} of {str(training_config)} and %{percentage*100} of target dataset")
            model = create_alexnet(num_classes=NUM_CLASSES)
            
            # Resize the training dataset to be equal to the size of the vietnam training set for consistency
            training_dataset = COVID19_Radiography(training_config)
            resized_training_dataset = data_utils.Subset(training_dataset, torch.arange(train_size))

            # Resize to the training dataset to the necessary portion for the test
            fractioned_training_dataset = data_utils.Subset(resized_training_dataset, torch.arange(int(train_size * (1-percentage))))
            
            # Resize the target testing dataset to the necessary portion for the test
            fractioned_vietnam_training_dataset = data_utils.Subset(vietnam_training_dataset, torch.arange(int(train_size * percentage)))

            # Concatenate the two datasets so that they are equal to the original target training dataset size
            mixed_training_dataset = torch.utils.data.ConcatDataset(
                [fractioned_training_dataset, fractioned_vietnam_training_dataset]
            )
            
            assert len(mixed_training_dataset) == train_size

            model_config = get_configuration_str(percentage, training_config, vietnam_dataset)
            model_path = os.path.join(MODELS_DIR_NAME, f"{model_config}.pt") 

            datasets = DatasetManager(mixed_training_dataset, vietnam_dataset)
            t = TrainAndTest(model, datasets, lr=BASE_LR, output_path="models/results.txt")
            report = t.train(NUM_EPOCHS)

            accuracy_matrix[i][j] = report['accuracy']
            precision_matrix[i][j] = report['weighted avg']['precision']
            recall_matrix[i][j] = report['weighted avg']['recall']
            f1_score_matrix[i][j] = report['weighted avg']['f1-score']

            with open("models/matrices.txt", "a") as f:
                f.write(f"Accuracy: {accuracy_matrix}\n")
                f.write(f"Precision: {precision_matrix}\n")
                f.write(f"Recall: {recall_matrix}\n")
                f.write(f"F1 Score: {f1_score_matrix}\n")

            # Save model and results
            t.save(model_config, model_path)
               

if __name__ == "__main__":
  main()