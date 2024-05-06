import torch
from torchvision import datasets, transforms 
from torchvision.models import resnet18, ResNet18_Weights
from tempfile import TemporaryDirectory
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.utils.data
import numpy as np
import os, time


# torch.utils.data.DataLoader()
DATA_DIR = "oxford-iiit-pet"
IMAGE_DATA_PATH = os.path.join(DATA_DIR, "images")
CATS_OR_DOGS = os.path.join(DATA_DIR, "cats-or-dogs")

# Use pre-trained ResNet18 model from torchvision
# Replace the last layer with 

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def main():
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  
    ])

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(512, 2)

    # Split the data into training, validation, and test sets
    # 70% train, 20% validation, 10% test
    train_dataset = datasets.ImageFolder(CATS_OR_DOGS, transform=train_transform)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size])

    print(len(train_dataset), len(val_dataset), len(test_dataset))

def train_model(model, criterion, optimizer, scheduler=0, num_epochs=25):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                #if phase == 'train':
                #    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model



# resnet = train_model(model, criterion, optimizer,
#                        num_epochs=20)

if __name__ == "__main__":
    main()

