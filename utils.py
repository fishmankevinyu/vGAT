import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
import os
import cv2
from einops import rearrange
class CardDataSet(Dataset):
    def __init__(self, card_frame, root_dir="C:\MyProjects\GAT_image\\", transform=None):
        self.card_frame = card_frame
        self.root_dir = root_dir
        self.transform = transform
        self.label_onehot, self.nClasses = encode_labels(card_frame['labels'])
        self.filepaths = card_frame['filepaths']
        self.filepaths = [os.path.join(self.root_dir, filepath.replace("/", "\\")) for filepath in self.filepaths]
    def __len__(self):
        return len(self.card_frame)
    def __getitem__(self, index):
        image = read_image(self.filepaths[index])
        if self.transform:
            image = self.transform(image)
        label = self.label_onehot[index]
        return {'image': image, 'label': label}

def encode_labels(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = [classes_dict[label] for label in labels]
    return labels, len(classes)

def load_data_card(path="C:\MyProjects\GAT_image\cards.csv"):
    # Load Cards Image Dataset
    card_frame = pd.read_csv(path)
    #print(card_frame)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CardDataSet(card_frame=card_frame.iloc[:7626], transform=transform)
    test_dataset = CardDataSet(card_frame=card_frame.iloc[7626: 7891], transform=transform)
    valid_dataset = CardDataSet(card_frame=card_frame.iloc[7891: 8156], transform=transform)

    patch_size = 16
    adj = np.zeros((patch_size * patch_size, patch_size * patch_size))

    for i in range(0, patch_size):
        for j in range(0, patch_size):
            index = i * patch_size + j
            adj[index, index] = 1
            if (i > 0):
                top = (i - 1) * patch_size + j
                adj[index, top] = 1
            if (i < patch_size - 1):
                bottom = (i + 1) * patch_size + j
                adj[index, bottom] = 1 
            if (j > 0):
                left = i * patch_size + j - 1
                adj[index, left] = 1
            if (j < patch_size - 1):
                right = i * patch_size + j + 1
                adj[index, right] = 1


    #normalize inputs
    #adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    #Tensor-lize inputs for cuda
    adj = torch.FloatTensor(np.array(adj))

    return adj, train_dataset, test_dataset, valid_dataset

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

