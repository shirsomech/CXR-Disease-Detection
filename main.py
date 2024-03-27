import torch
import torch.utils.data as data_utils
from torchvision import models
from CXRDataset import COVID19_Radiography, VinXRay
from config import BASE_LR, NUM_EPOCHS, TARGET_SITE_PERCENTAGE_IN_TESTING_DATASET
from TrainAndTest import TrainAndTest
from DatasetManager import DatasetManager
import numpy as np

def create_alexnet(in_channels):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    return model

def main():
    alexnet = create_alexnet(in_channels=1)
    
    # Baseline
    vietnam_dataset = VinXRay()
    
    train_size = int(0.8 * len(vietnam_dataset))
    test_size = len(vietnam_dataset) - train_size
    vietnam_training_dataset, vietnam_testing_dataset = torch.utils.data.random_split(vietnam_dataset, [train_size, test_size])

    datasets = DatasetManager(vietnam_training_dataset, vietnam_testing_dataset)
    t = TrainAndTest(alexnet, datasets, lr=BASE_LR)

    #confusion_matrix = t.train(NUM_EPOCHS)

    # Building the testing configuration matrix
    #cxr_datasets_config = [
    #    COVID19_Radiography.Datasets.NORMAL_DS1,
    #    COVID19_Radiography.Datasets.NORMAL_DS2,
    #    COVID19_Radiography.Datasets.COVID_DS3,
    #    COVID19_Radiography.Datasets.COVID_DS4,
    #    COVID19_Radiography.Datasets.COVID_DS5,
    #    COVID19_Radiography.Datasets.COVID_DS6,
    #    COVID19_Radiography.Datasets.COVID_DS7,
    #    COVID19_Radiography.Datasets.COVID_DS8,
    #    COVID19_Radiography.Datasets.LO_DS1,
    #    COVID19_Radiography.Datasets.VP_DS2
    #]

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

            print(f"Training on %{(1-percentage)*100} of {str(training_config)} and %{percentage*100} of target dataset")
            
            datasets = DatasetManager(training_config_matrix[i][j], vietnam_testing_dataset)
            t = TrainAndTest(alexnet, datasets, lr=BASE_LR)
            accuracies, losses = t.train(NUM_EPOCHS)

            for split in ['train', 'val']:
                print(split, "accuracies by epoch:", accuracies[split])
                print(split, "losses by epoch:", losses[split])

    # state = torch.load('fine_tuned_best_model.pt')
    # model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])

    # model_ft.load_state_dict(torch.load('fine_tuned_best_model.pt')) 

    # Run the functions and save the best model in the function model_ft.
   
    # Save model
    # torch.save(model_ft.state_dict(), 'alexnet_model.pt')


if __name__ == "__main__":
  main()