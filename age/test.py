import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import tqdm
import pandas as pd
import matplotlib.pyplot as plt 

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

class AgeDataset(Dataset):
    def __init__(self, young_folder_path, adult_folder_path, old_folder_path):
        self.timesteps = []
        self.labels = []
        self.all_files = []
        
        for img_filenames in os.listdir(young_folder_path):
            for filename in os.listdir(os.path.join(young_folder_path, img_filenames)):
                data = os.path.join(young_folder_path, img_filenames, filename)
                number_str = filename.split('_')[1]
                step = int(number_str.split('.')[0])
                self.timesteps.append(step)
                self.labels.append(0)
                self.all_files.append(data)
                
        for img_filenames in os.listdir(adult_folder_path):
            for filename in os.listdir(os.path.join(adult_folder_path, img_filenames)):
                data = os.path.join(adult_folder_path, img_filenames, filename)
                number_str = filename.split('_')[1]
                step = int(number_str.split('.')[0])
                self.timesteps.append(step)
                self.labels.append(1)
                self.all_files.append(data)
        
        for img_filenames in os.listdir(old_folder_path):
            for filename in os.listdir(os.path.join(old_folder_path, img_filenames)):
                data = os.path.join(old_folder_path, img_filenames, filename)
                number_str = filename.split('_')[1]
                step = int(number_str.split('.')[0])
                self.timesteps.append(step)
                self.labels.append(2)
                self.all_files.append(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        latent = torch.load(self.all_files[idx])
        label = self.labels[idx]
        timestep = self.timesteps[idx]
        return latent, label, timestep

print("split train and test")
dataset = AgeDataset('data/young_test_outputs', 'data/adult_test_outputs', 'data/old_test_outputs')

test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("build model")
model = Model(dim=3).half()
model.load_state_dict(torch.load('./model_pt/best_model_weights.pth'))
model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)

step_accuracy = {}

print("test acc")
with torch.no_grad():
    for i, (latent, label, timestep) in enumerate(tqdm.tqdm(test_loader)):
        latent = latent.to('cuda')
        label = label.to('cuda')
        timestep = timestep.to('cuda')
        
        output = model(latent, timestep)
        loss = criterion(output, label)
        
        acc = (output.argmax(1) == label).sum().item()
        step = timestep.item()
        if step not in step_accuracy:
            step_accuracy[step] = {'correct': 0, 'total': 0}
            
        step_accuracy[step]['correct'] += acc
        step_accuracy[step]['total'] += 1

for step in step_accuracy:
    step_accuracy[step]['accuracy'] = step_accuracy[step]['correct'] / step_accuracy[step]['total']
    
steps = sorted(step_accuracy.keys())
accuracies = [step_accuracy[step]['accuracy'] for step in steps]

plt.figure(figsize=(10, 5))
plt.plot(steps, accuracies, marker='o')
plt.title('Age Step-wise Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.xticks(steps)  
plt.grid()
plt.show()
plt.savefig('age_accuracy_step.png')  
print("图像已保存为 age_accuracy_step.png")

accuracy_df = pd.DataFrame.from_dict(step_accuracy, orient='index')
accuracy_df.reset_index(inplace=True)
accuracy_df.rename(columns={'index': 'Step'}, inplace=True)

accuracy_df.to_csv('age_step_accuracy.csv', index=False)