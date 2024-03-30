import torch
import torch.utils.data as data_utils
import torch.nn as nn
from torchvision import models
from CXRDataset import COVID19_Radiography, VinXRay
from config import BASE_LR, NUM_EPOCHS, TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET, NUM_CLASSES, MODELS_DIR_NAME
from TrainAndTest import TrainAndTest
from DatasetManager import DatasetManager
import numpy as np

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
    confusion_matrix = t.train(NUM_EPOCHS)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR_NAME, "baseline.pt"))

    # Building the testing configuration matrix
    cxr_datasets_config = [
        COVID19_Radiography.Datasets.NORMAL_DS1,
        COVID19_Radiography.Datasets.NORMAL_DS2,
        COVID19_Radiography.Datasets.COVID_DS3,
        COVID19_Radiography.Datasets.COVID_DS4,
        COVID19_Radiography.Datasets.COVID_DS5,
        COVID19_Radiography.Datasets.COVID_DS6,
        COVID19_Radiography.Datasets.COVID_DS7,
        COVID19_Radiography.Datasets.COVID_DS8,
        COVID19_Radiography.Datasets.LO_DS1,
        COVID19_Radiography.Datasets.VP_DS2
    ]

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
    training_dataset_configs = [one_dataset, two_datasets, four_datasets, six_datasets, eight_datasets]

    training_config_matrix = np.empty((len(training_dataset_configs), len(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET)), dtype=data_utils.ConcatDataset)

    for i, training_config in enumerate(training_dataset_configs):
        for j, percentage in enumerate(TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET):
            original_training_dataset = COVID19_Radiography(training_config)

            cxr_size = int((1 - percentage) * len(original_training_dataset))
            resized_original_training_dataset = data_utils.Subset(original_training_dataset, torch.arange(cxr_size))
            
            vietnam_size = int(percentage * len(original_training_dataset))
            resized_vietnam_training_dataset = data_utils.Subset(vietnam_training_dataset, torch.arange(vietnam_size))

            training_config_matrix[i][j] = torch.utils.data.ConcatDataset(
                [resized_original_training_dataset, resized_vietnam_training_dataset]
            )
            
            #TODO: Resize original training set to a set number 
            print(f"Training on %{(1-percentage)*100} of {str(training_config)} and %{percentage*100} of target dataset")
            
            datasets = DatasetManager(training_config_matrix[i][j], vietnam_dataset)
            model = create_alexnet(num_classes=NUM_CLASSES)
            t = TrainAndTest(model, datasets, lr=BASE_LR)
            accuracies, losses = t.train(NUM_EPOCHS)

            for split in ['train', 'val']:
                print(split, "accuracies by epoch:", accuracies[split])
                print(split, "losses by epoch:", losses[split])

            # Save model
            model_config = get_configuration_str(percentage, training_config, vietnam_dataset)
            model_path = os.path.join(MODELS_DIR_NAME, "{model_config}.pt")
            torch.save(model_ft.state_dict(), model_path)
               

if __name__ == "__main__":
  main()