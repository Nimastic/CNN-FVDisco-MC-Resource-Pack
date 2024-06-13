import torch
import torch.nn as nn
from torchvision import transforms
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('model.ckpt'))
model.eval()

input_dir = './dataset/trainA'
output_dir = './dataset/generatedB'
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

with torch.no_grad():
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png') and file_name not in os.listdir('./dataset/trainB'):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            output = model(img).squeeze(0).cpu()

            output_img = transforms.ToPILImage()(output)
            output_img.save(os.path.join(output_dir, file_name))

print("Image generation completed.")
