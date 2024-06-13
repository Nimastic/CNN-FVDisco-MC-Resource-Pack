import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PairedDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None):
        self.rootA = rootA
        self.rootB = rootB
        self.transform = transform
        self.filesA = sorted([f for f in os.listdir(rootA) if f.endswith('.png')])
        self.filesB = sorted([f for f in os.listdir(rootB) if f.endswith('.png')])
        # Find common files
        self.common_files = list(set(self.filesA) & set(self.filesB))

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, index):
        file_name = self.common_files[index]
        imgA = Image.open(os.path.join(self.rootA, file_name)).convert('RGB')
        imgB = Image.open(os.path.join(self.rootB, file_name)).convert('RGB')
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = PairedDataset('./dataset/trainA', './dataset/trainB', transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

epochs = 50
for epoch in range(epochs):
    for i, (imgA, imgB) in enumerate(train_loader):
        imgA, imgB = imgA.to(device), imgB.to(device)

        outputs = model(imgA)
        loss = criterion(outputs, imgB)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.ckpt')
