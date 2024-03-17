import torch
from torchvision import models
from CXRDataset import COVID19_Radiography, COVID19_RadiographyDatasets
from config import BASE_LR, NUM_EPOCHS
from TrainAndTest import TrainAndTest

def create_alexnet(in_channels):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    return model

def main():

    config = [
        COVID19_RadiographyDatasets.NORMAL_1,
        COVID19_RadiographyDatasets.NORMAL_2,
        COVID19_RadiographyDatasets.COVID_1,
        COVID19_RadiographyDatasets.COVID_2,
        COVID19_RadiographyDatasets.COVID_3,
        COVID19_RadiographyDatasets.COVID_4,
        COVID19_RadiographyDatasets.COVID_5,
        COVID19_RadiographyDatasets.COVID_6,
        COVID19_RadiographyDatasets.LO_1,
        COVID19_RadiographyDatasets.VP_1
    ]

    cxr_dataset = COVID19_Radiography("datasets/COVID-19_Radiography.zip", config)
    
    train_size = int(0.8 * len(cxr_dataset))
    test_size = len(cxr_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(cxr_dataset, [train_size, test_size])

    # state = torch.load('fine_tuned_best_model.pt')
    # model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])

    # model_ft.load_state_dict(torch.load('fine_tuned_best_model.pt')) 

    alexnet = create_alexnet(in_channels=1)
    
    # Run the functions and save the best model in the function model_ft.
    t = TrainAndTest(alexnet, train_dataset, test_dataset, lr=BASE_LR)
    accuracies, losses = t.train(NUM_EPOCHS)

    for split in ['train', 'val']:
        print(split, "accuracies by epoch:", accuracies[split])
        print(split, "losses by epoch:", losses[split])

    # Save model
    #torch.save(model_ft.state_dict(), 'alexnet_model.pt')


if __name__ == "__main__":
  main()