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
    plt.show()

def main():
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Split the data into training, validation, and test sets
    # 70% train, 20% validation, 10% test
    train_dataset = datasets.ImageFolder(CATS_OR_DOGS, transform=train_transform)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
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

    test_model(resnet, dataloaders['test'], 'best_model_params.pt')

    # train_model(resnet, dataloaders, dataset_sizes)

def test_model(model, test_data, weights_path, device_type='mps'):
    device = torch.device(device_type)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            labels = labels.unsqueeze(1).float()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct / total}%')


def train_model(model, dataloaders, dataset_sizes, scheduler=0, num_epochs=25):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    since = time.time()

    # Create a directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join("./", 'best_model_params.pt')

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
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) # forward pass
                        # make labels one dimension higher
                        labels = labels.unsqueeze(1).float()
                        preds = (outputs > 0.5).float()

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

def train2(model, optimizer, loss_fn, trn_dl, val_dl):
    train_losses, train_accuracies = [], []
    val_accuracies = []

    print("All losses and accuracies are for each epoch")
    for epoch in range(5):
        
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss) 
        train_epoch_loss = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        print(f" epoch {epoch + 1}/5, Training Loss: {train_epoch_loss}, Training Accuracy: {train_epoch_accuracy}, Validation Accuracy: {val_epoch_accuracy}")
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)


if __name__ == "__main__":
    main()

