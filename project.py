from torchvision import datasets, transforms 
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from tempfile import TemporaryDirectory
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.utils.data
import numpy as np
import os, time
from torchvision.datasets import OxfordIIITPet

# torch.utils.data.DataLoader()
DATA_DIR = "oxford-iiit-pet"
IMAGE_DATA_PATH = os.path.join(DATA_DIR, "images")
CATS_OR_DOGS = os.path.join(DATA_DIR, "cats-or-dogs")

# Use pre-trained ResNet18 model from torchvision
# Replace the last layer with 

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

def test_model(model, test_data, weights_path, device_type='cpu'):
    if torch.cuda.is_available():
        device_type = 'cuda:0'
    elif torch.backends.mps.is_available():
        device_type = 'mps'

    device = torch.device(device_type)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)

    start = time.time()
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(outputs, 1)

            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
        print('Time taken:', time.time() - start)
        print(f'Accuracy of the network on the test images: {100 * correct / total}%')


def train_model(model, dataloaders, dataset_sizes, suffix, scheduler=0, num_epochs=25, device_type='cpu'):
    if torch.cuda.is_available():
        device_type = 'cuda:0'
    elif torch.backends.mps.is_available():
        device_type = 'mps'

    device = torch.device(device_type)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    since = time.time()

    # Create a directory to save training checkpoints
    with TemporaryDirectory() as tempdir, open('outputs/training-out-' + suffix + '.txt', 'w') as f:
        best_model_params_path = os.path.join("./", suffix + '.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        start = time.time()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                print('phase:', phase)
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) # forward pass

                        # outputs is probability of each class
                        # labels needs to be probability of each class

                        # we need the one-hot for the labels to get the loss
                        true_outputs = torch.nn.functional.one_hot(labels, num_classes=37).float()

                        # create preds by thresholding outputs
                        loss = criterion(outputs, true_outputs)

                        preds = torch.argmax(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                outputstr = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
                print(outputstr)
                f.write(outputstr + '\n')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            print('epoch took:', time.time() - start)
            start = time.time()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def generate_datasets_binary(
        # 70% train, 20% validation, 10% test
        train_split=0.7, val_split=0.2, crop=True, random_flip=False
    ):

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if crop:
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))
    else:
        transform_list.append(transforms.Resize(224))

    train_transform = transforms.Compose(
        transform_list
    )

    # Split the data into training, validation, and test sets
    train_size = int(train_split * len(train_dataset))
    val_size = int(val_split * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True),
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset),
        'val': len(val_dataset),
    }

    return dataloaders, dataset_sizes

def generate_datasets(
        # 70% train, 20% validation, 10% test
        train_split=0.7, val_split=0.2, crop=True, random_flip=False,
    ):

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if crop:
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))
    else:
        transform_list.append(transforms.Resize(224))

    train_transform = transforms.Compose(
        transform_list
    )

    dataset=OxfordIIITPet(root="./", download=True, target_types='category', transform=train_transform)

    # Split the data into training, validation, and test sets
    train_dataset = dataset
    train_size = int(train_split * len(train_dataset))
    val_size = int(val_split * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True),
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset),
        'val': len(val_dataset),
    }

    return dataloaders, dataset_sizes

def binary_resnet():
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in resnet.parameters():
        param.requires_grad = False

    # resnet.fc = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(512, 128),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(128, 1),
    #     nn.Sigmoid()
    # )

    # Modify the last layer of the model

    # fc = Final fully connected layer
    resnet.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    
    return resnet

def multiclass_resnet(resnet_size="18"):
    if resnet_size == "18":
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif resnet_size == "50":
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in resnet.parameters():
        param.requires_grad = False
    
    print(resnet.fc)

    #resnet fc input size
    input_size = resnet.fc.in_features

    resnet.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 256),
        nn.ReLU(),
        # nn.Dropout(0.2),
        nn.Linear(256, 37),
        nn.Softmax()
    )

    return resnet

def create_suffix(
        crop=True, random_flip=False,
        resnet_size="18",
    ):
    suffix = ""
    if crop:
        suffix += "crop"
    if random_flip:
        suffix += "flip"
    suffix += resnet_size
    return suffix


def main():
    # for binary
    # dataset = datasets.ImageFolder(CATS_OR_DOGS, transform=train_transform)

    dataloaders, dataset_sizes = generate_datasets()

    resnet_size = "50"

    resnet = multiclass_resnet(
        resnet_size=resnet_size
    )
    suffix = create_suffix(
        resnet_size=resnet_size
    )

    print(suffix)

    # train_model(resnet, dataloaders, dataset_sizes, suffix + '-cpu-test', num_epochs=2)

    weights_path = "./" + suffix + '-cpu-test' + ".pt"
    # weights_path = "./best_model_params_multi.pt"

    test_model(resnet, dataloaders['test'], weights_path, device_type='cpu')

if __name__ == "__main__":
    main()

