# Project Overview
This project aims to create a neural network model that transfers the art style of FVDisco Minecraft texture pack to the default Minecraft texture pack. Using Convolutional Neural Networks (CNNs), we train a model on a paired dataset of textures and use it to generate the remaining fvdisco-styled textures.

```
project_root/
├── dataset/
│   ├── trainA/
│   │   ├── ... (default Minecraft textures)
│   ├── trainB/
│   │   ├── ... (fvdisco textures)
│   └── generatedB/
│       ├── ... (generated fvdisco textures)
├── train_cnn.py
├── generate_images.py
├── replace_pngs.py
├── README.md
└── ...

```

```pip install torch torchvision pillow numpy```

## Dataset Preparation
Place the default Minecraft textures in dataset/trainA.
Place the fvdisco textures in dataset/trainB.
Ensure that the filenames in trainA and trainB match for the textures you have.

Run the training script: ```python train_cnn.py```
Run the generation script: ```python generate_images.py```
Run the replacement script: ```python replace_pngs.py```

## CNN Architecture
Here is an overview of the CNN architecture used for the style transfer:
```
Input (3x256x256)
   |
   V
+-----------------------------+
| Conv2d(3, 64, kernel_size=3, |----> Output: 64x256x256
| stride=1, padding=1)        |
+-----------------------------+
          ReLU
           |
           V
+-----------------------------+
| Conv2d(64, 128, kernel_size=3|----> Output: 128x128x128
| stride=2, padding=1)        |
+-----------------------------+
          ReLU
           |
           V
+-----------------------------+
| Conv2d(128, 256, kernel_size=3|---> Output: 256x64x64
| stride=2, padding=1)        |
+-----------------------------+
          ReLU
           |
           V
+-----------------------------+
| ConvTranspose2d(256, 128,   |----> Output: 128x128x128
| kernel_size=3, stride=2,    |
| padding=1, output_padding=1)|
+-----------------------------+
          ReLU
           |
           V
+-----------------------------+
| ConvTranspose2d(128, 64,    |----> Output: 64x256x256
| kernel_size=3, stride=2,    |
| padding=1, output_padding=1)|
+-----------------------------+
          ReLU
           |
           V
+-----------------------------+
| ConvTranspose2d(64, 3,      |----> Output: 3x256x256
| kernel_size=3, stride=1,    |
| padding=1)                  |
+-----------------------------+
           Tanh
           |
           V
Output (3x256x256)

```

## Explanation of the CNN Architecture
The architecture of the Convolutional Neural Network (CNN) used for style transfer in this project is designed to effectively capture and replicate the stylistic features of the fvdisco texture pack onto the default Minecraft textures. Here's a detailed breakdown of the architecture and the rationale behind its design:

### Encoder-Decoder Structure
The chosen architecture is an encoder-decoder network, which is commonly used for tasks involving image-to-image translation. The encoder compresses the input image into a lower-dimensional representation, capturing the essential features. The decoder then reconstructs the image from this representation, applying the learned stylistic features.

##  Detailed Architecture
1. Input Layer (3x256x256):
The input is a 256x256 RGB image with 3 color channels.

2. Convolutional Layers (Encoder):
Conv2d(3, 64, kernel_size=3, stride=1, padding=1) -> ReLU:
- This layer applies 64 filters of size 3x3 with a stride of 1 and padding of 1, resulting in an output of the same spatial dimensions (256x256). The ReLU activation introduces non-linearity.
Conv2d(64, 128, kernel_size=3, stride=2, padding=1) -> ReLU:
- This layer applies 128 filters of size 3x3 with a stride of 2 and padding of 1, reducing the spatial dimensions by half (128x128). The ReLU activation follows.
Conv2d(128, 256, kernel_size=3, stride=2, padding=1) -> ReLU:
- This layer applies 256 filters of size 3x3 with a stride of 2 and padding of 1, further reducing the spatial dimensions by half (64x64). The ReLU activation follows.

3. Transposed Convolutional Layers (Decoder):
ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) -> ReLU:
- This layer applies 128 transposed filters of size 3x3 with a stride of 2, padding of 1, and output padding of 1, increasing the spatial dimensions to 128x128. The ReLU activation follows.
ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) -> ReLU:
- This layer applies 64 transposed filters of size 3x3 with a stride of 2, padding of 1, and output padding of 1, increasing the spatial dimensions to 256x256. The ReLU activation follows.
ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1) -> Tanh:
- This layer applies 3 transposed filters of size 3x3 with a stride of 1 and padding of 1, resulting in an output image of the same spatial dimensions (256x256). The Tanh activation ensures the output pixel values are in the range [-1, 1], which is suitable for image data.

## Why This Architecture?
Feature Extraction and Compression:

The encoder part of the network progressively reduces the spatial dimensions while increasing the depth (number of filters), which allows the network to capture complex features at different levels of abstraction.
Reconstruction with Style Transfer:

The decoder part of the network reconstructs the image from the compressed representation. During training, the network learns to apply the stylistic features of trainB (fvdisco textures) while maintaining the structural integrity of trainA (default Minecraft textures).
Non-linearity and Complex Patterns:

The use of ReLU activations introduces non-linearity, allowing the network to learn complex patterns and relationships in the data.
High-Quality Resampling:

The transposed convolutional layers (also known as deconvolutional layers) upsample the feature maps back to the original image size, ensuring that the stylistic details are preserved and applied correctly.
Output Range Adjustment:

The Tanh activation function in the final layer ensures that the output pixel values are in a suitable range for image data, facilitating a smooth transition in the learned style.


