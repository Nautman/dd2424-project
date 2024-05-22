import torch
from torch.utils.data import Dataset, random_split, DataLoader

# Create a custom dataset of numbers from 1 to 100
class NumberDataset(Dataset):
    def __init__(self):
        self.data = list(range(1, 101))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Function to perform the split
def perform_split(dummy_variable):
    dataset = NumberDataset()
    dataset_length = len(dataset)
    
    # Set seed based on dummy_variable
    torch.manual_seed(42)
    
    # Define lengths for the split
    lengths = [dataset_length // 2, dataset_length - dataset_length // 2]
    
    # Perform random split
    split1, split2 = random_split(dataset, lengths)
    
    # Return the two splits as lists of numbers
    return list(split1), list(split2)

# Main script
if __name__ == "__main__":
    dummy_variable = 2  # Change this to 2 for the second run
    
    split1, split2 = perform_split(dummy_variable)
    
    print("First split:", [item for item in split1])
    print("Second split:", [item for item in split2])