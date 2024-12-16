import re
import os
import torch
import pickle
import logging
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader



# Implements dataset file saving and loading
def saveDatasetToFile(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def loadDatasetFromFile(filename):
    with open(filename, 'rb') as f:
        loaded_dataset = pickle.load(f)
    return loaded_dataset



# Custom dataset class for Breast Cancer Datasets
class BreastCancerDataset(Dataset):
    def __init__(self, data, folder_path, label, transform=None):
        self.data = data
        self.folder_path = folder_path
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, mask_name = self.data[idx]
        image = Image.open(os.path.join(self.folder_path, img_name)).convert('L')
        mask = Image.open(os.path.join(self.folder_path, mask_name)).convert('L')
        label = self.label
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return (image, mask, label)



# Implements creation of datasets from their respective folders
def createDataset(folder_path, label, transform=None, mask_transform=None):

    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    data = process_data(image_files)
    return BreastCancerDataset(data, folder_path, label, transform)



# Splits the image and mask filename and then processes it
def process_data(image_files, ext='.png'):
    data = []
    for image_file in image_files:
        f_split = re.split('[._]', image_file)
        if "mask" not in f_split:
            img_name = f_split[0] + ext
            mask_name = f_split[0] + "_mask" + ext
            data.append((img_name, mask_name))
    return data



def make_data(ROOT_DIR, args):
    
    # Define image class folders
    benign_folder = os.path.join(ROOT_DIR, "benign") 
    malignant_folder = os.path.join(ROOT_DIR, "malignant") 
    normal_folder = os.path.join(ROOT_DIR, "normal")


    # Define parameters
    IMAGE_SIZE = 128
    BATCH_SIZE = args.batch_size


    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Create dataset according to classes
    normal_dataset = createDataset(normal_folder, 0, transform=transform)
    benign_dataset = createDataset(benign_folder, 1, transform=transform)
    malignant_dataset = createDataset(malignant_folder, 2, transform=transform)

    logging.info(f'''Breastcancer dataset lengths:
                    Normal dataset length:      {len(normal_dataset)}
                    Benign dataset length:      {len(benign_dataset)}
                    Malignant dataset length:   {len(malignant_dataset)}
                 ''')

    # Combine the three datasets into a single dataset
    combined_dataset = ConcatDataset([normal_dataset, benign_dataset, malignant_dataset])

    # Define the sizes for validation, and test splits
    valid_size = args.val_size
    test_size = args.test_size

    # Shuffle the combined dataset using a random seed
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    shuffled_indices = torch.randperm(len(combined_dataset))

    # Split the shuffled dataset into train, validation, and test datasets
    train_indices, temp_indices = train_test_split(shuffled_indices, test_size=valid_size + test_size, random_state=random_seed)
    valid_test_indices = temp_indices.tolist()
    valid_indices, test_indices = train_test_split(valid_test_indices, test_size=test_size / (valid_size + test_size), random_state=random_seed)

    # Prepare datasets for train, valid and test datasets
    train_dataset = Subset(combined_dataset, train_indices)
    valid_dataset = Subset(combined_dataset, valid_indices)
    test_dataset = Subset(combined_dataset, test_indices)
    
    logging.info(f'''Total samples:
                    Training:      {len(train_dataset)}
                    Valid:         {len(valid_dataset)}
                    Test:          {len(test_dataset)}
                 ''')

    # Prepare Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader