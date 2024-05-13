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
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
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
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size])

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

    # for i in range(20):
    #     show(dataloaders['train'].dataset[i][0])
    #     print(dataloaders['train'].dataset[i][1])

    # print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_model(resnet, dataloaders, dataset_sizes)
    # optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    # loss_fn = nn.BCELoss()
    # train2(resnet, optimizer, loss_fn, dataloaders['train'], dataloaders['val'])


def train_model(model, dataloaders, dataset_sizes, scheduler=0, num_epochs=25):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
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
                        outputs = model(inputs) # forward pass
                        # make labels one dimension higher
                        labels = labels.unsqueeze(1).float()
                        preds = (outputs > 0.5).float()

                        loss = criterion(outputs, labels)

                        print('preds', preds)
                        print('labels.data', labels.data)

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

