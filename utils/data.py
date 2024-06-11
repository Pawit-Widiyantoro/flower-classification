import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from glob import glob

class Data(Dataset):
    def __init__(self, folder="C:/Users/Lenovo/Documents/Kuliah/Semester 6/Pengenalan Pola/flower-classification/dataset/flowers/"):
        self.dataset = []

        for daisy in glob(folder + "daisy/*"):
            image = cv.imread(daisy)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([image / 255.0, 0])

        for dandelion in glob(folder + "dandelion/*"):
            image = cv.imread(dandelion)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([image / 255.0, 1])

        for rose in glob(folder + "rose/*"):
            image = cv.imread(rose)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([image / 255.0, 2])

        for sunflower in glob(folder + "sunflower/*"):
            image = cv.imread(sunflower)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([image / 255.0, 3])

        for tulip in glob(folder + "tulip/*"):
            image = cv.imread(tulip)
            image = cv.resize(image, (100, 100))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.dataset.append([image / 255.0, 4])

        # Convert the dataset to numpy arrays for faster tensor creation
        self.features = np.array([item[0] for item in self.dataset])
        self.labels = np.array([item[1] for item in self.dataset])

    def __getitem__(self, item):
        feature = torch.tensor(self.features[item], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[item], dtype=torch.int64)
        return feature, label

    def __len__(self):
        return len(self.dataset)
    
if __name__=="__main__":
    data = Data()