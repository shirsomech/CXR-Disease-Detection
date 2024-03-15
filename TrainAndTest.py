from __future__ import print_function, division
from DatasetManager import DatasetManager
from config import BASE_LR, EPOCH_DECAY, DECAY_WEIGHT
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.autograd import Variable

def create_resnet(in_channels):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    return model 

class TrainAndTest(object):
    def __init__(self, model, training_dataset, test_dataset, config=None, lr=BASE_LR):
        self.model_ft = model
        self.dataset_manager = DatasetManager(training_dataset, test_dataset)
        self.accuracies = []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.model_ft.parameters(), lr=BASE_LR)
        self.phases = ['Train', 'Val']

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in self.phases:
                if phase == 'Train':
                    self.optimizer = self.__lr_scheduler(epoch)
                    self.model_ft.train()
                else:
                    self.model_ft.eval() 

                running_loss = 0.0
                running_corrects = 0
                counter = 0
                for data in self.dataset_manager.dataloaders[phase]:
                    inputs, labels = data
                    inputs, labels = Variable(inputs), Variable(labels)

                    self.optimizer.zero_grad()
                    outputs = self.model_ft(inputs)
                    _, preds = torch.max(outputs.data, 1)

                    loss = self.criterion(outputs, labels)

                    # Print a line every 10 batches so you have something to watch and don't feel like the program isn't running.
                    if counter % 10 == 0:
                        print("Reached batch iteration", counter)

                    counter += 1


                    if phase == 'Train':
                        loss.backward()
                        self.optimizer.step()
                    try:
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds == labels.data)
                    except:
                        print('unexpected error, could not calculate loss or do a sum.')

                epoch_loss = running_loss / self.dataset_manager.dataset_sizes[phase]
                epoch_acc = running_corrects.item() / float(self.dataset_manager.dataset_sizes[phase])
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                accuracies[phase].append(epoch_acc)
                losses[phase].append(epoch_loss)

                # deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(model)
                        print('new best accuracy =', best_acc)
        
        time_elapsed = time.time() - since
        
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('returning and looping back')

        return accuracies, losses


    def __lr_scheduler(self, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
        lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return self.optimizer

    def get_results():
        return self.accuracies, self.losses
    
    @property
    def model():
        return self.model

