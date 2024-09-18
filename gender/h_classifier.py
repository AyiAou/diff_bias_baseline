import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import tqdm
import pandas as pd

class Model(nn.Module):
    def __init__(self, dim=2):
        super(Model, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(2 * 1280 * 8 * 8, dim) for i in range(48)])

    def forward(self, x, t):
        x = x.reshape(x.shape[0], -1)
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(self.linears[t[i] - 1](x[i])) 
        return torch.stack(outputs)

class GenderDataset(Dataset):
    def __init__(self, male_folder_path, female_folder_path):
        self.timesteps = []
        self.labels = []
        self.all_files = []
        for img_filenames in os.listdir(male_folder_path):
            for filename in os.listdir(os.path.join(male_folder_path, img_filenames)):
                number_str = filename.split('_')[1]
                step = int(number_str.split('.')[0])
                self.timesteps.append(step)
                self.labels.append(0)
                self.all_files.append(os.path.join(male_folder_path, img_filenames, filename))
        for img_filenames in os.listdir(female_folder_path):
            for filename in os.listdir(os.path.join(female_folder_path, img_filenames)):
                number_str = filename.split('_')[1]
                step = int(number_str.split('.')[0])
                self.timesteps.append(step)
                self.labels.append(1)
                self.all_files.append(os.path.join(female_folder_path, img_filenames, filename))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        latent = torch.load(self.all_files[idx])
        label = self.labels[idx]
        timestep = self.timesteps[idx]
        return latent, label, timestep


def fast_train_test_split(dataset, test_size=0.1, random_state=None):
    num_samples = len(dataset)
    num_test = int(num_samples * test_size)
    indices = np.arange(num_samples)

    # np.random.seed(random_state)
    np.random.shuffle(indices)

    train_indices = indices[num_test:]
    test_indices = indices[:num_test]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

print("split train and test")
dataset = GenderDataset('./autodl-tmp/male_outputs', './autodl-tmp/female_outputs')
train_dataset, test_dataset = fast_train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("build model")
model = Model().half()
model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)

best_loss = float('inf')
best_model_weights = None
last_model_weights = None
train_accuracy_history = []
test_accuracy_history = []

print("train model")
for epoch in range(10):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (latent, label, timestep) in enumerate(tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        latent = latent.to('cuda')
        label = label.to('cuda')
        timestep = timestep.to('cuda')
        output = model(latent, timestep)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (output.argmax(1) == label).sum().item()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}: train loss {train_loss:.4f}, train acc {train_acc:.4f}')

    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for i, (latent, label, timestep) in enumerate(tqdm.tqdm(test_loader)):
            latent = latent.to('cuda')
            label = label.to('cuda')
            timestep = timestep.to('cuda')
            output = model(latent, timestep)
            loss = criterion(output, label)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == label).sum().item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    print(f'Epoch {epoch+1}: test loss {test_loss:.4f}, test acc {test_acc:.4f}')

    if test_loss < best_loss:
        best_loss = test_loss
        best_model_weights = model.state_dict()
    if epoch == 9:
        last_model_weights = model.state_dict()
        torch.save(last_model_weights, './autodl-tmp/model_pt/last_model_weights.pth')
    train_accuracy_history.append(train_acc)
    test_accuracy_history.append(test_acc)

torch.save(best_model_weights, './autodl-tmp/model_pt/best_model_weights.pth')
accuracy_df1 = pd.DataFrame(train_accuracy_history, columns=['Accuracy'])
accuracy_df1.to_csv('./autodl-tmp/train_accuracy_history.csv', index=False)
accuracy_df2 = pd.DataFrame(test_accuracy_history, columns=['Accuracy'])
accuracy_df2.to_csv('./autodl-tmp/test_accuracy_history.csv', index=False)