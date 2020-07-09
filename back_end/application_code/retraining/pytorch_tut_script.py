import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# define data dir
data_dir = "data/hymenoptera_data"

# start with this squeezenet
model_name = "squeezenet"

# number of classes in dataset
num_classes = 2

# batch size
batch_size = 8

# number epochs
num_epochs = 15

# flat for feature extract
feature_extract = True

# input size
input_size = 224

def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # training and validation phase for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # special case for inception because in training it has an auxiliary output. blabla

                    if is_inception and phase == "train":

                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backwards + optimize only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # keep track of stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # make deep copy of model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.9f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def re_initialize_model():

    # Load model
    squeeze = models.squeezenet1_0(pretrained=True)
    print(squeeze)

    # Freeze grads
    set_parameter_requires_grad(squeeze, feature_extract)

    # Reshape relevant layer
    squeeze.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

    # Set some other params
    squeeze.num_classes = num_classes
    input_size = 224

    return squeeze, input_size


def load_data():

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])

        ]),
        'val' : transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ]),
    }

    print("Initialising Datasets and Dataloaders")

    # create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(os.path.join(os.getcwd(), data_dir), x), data_transforms[x]) for x in ['train', 'val']}

    # create training and validation loaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # detect if we have gpu available
    device = torch.device('cudo:0' if torch.cuda.is_available() else "cpu")

    return dataloaders_dict, device

def set_param_model(model, device):

    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer

def retrain_model(model, dataloaders_dict, optimizer):

    # set up loss function
    criterion = nn.CrossEntropyLoss()

    # train and eval
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer,
            num_epochs=num_epochs, is_inception=False)


if __name__ == '__main__':

    model, input_size = re_initialize_model()
    dataloaders_dict, device = load_data()
    optimizer = set_param_model(model, device)
    retrain_model(model, dataloaders_dict, optimizer)


