# CIFAR10 MLP vs CNN Comparison

This project compares two machine learning models for classifying the CIFAR10 dataset:
- **MLP (Multi-Layer Perceptron)**
- **CNN (Convolutional Neural Network)**

## File Structure

CIAF10-MLP-vs-CNN/

├── Cifar-10_CNN.ipynb

├── CIFAR-10_dataset.jpg

├── Cifar-10_using_1HiddenLayer.ipynb

├── Cifar-10_using_2HiddenLayer.ipynb

├── Cifar-10_using_3HiddenLayer.ipynb

└── README.md

- `Cifar-10_CNN.ipynb`: A Notebook for training a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.
- `CIFAR-10_dataset.jpg`: The CIFAR10 dataset (Color images, Resolution=32x32, Training set: 50000 samples, Testing set: 10000 samples).
- `Cifar-10_using_1HiddenLayer.ipynb`: A Notebook for training a Multi-Layer Perception (MLP) using 1 Hidden Layer for image classification on the CIFAR-10 dataset.
- `Cifar-10_using_2HiddenLayer.ipynb`: A Notebook for training a Multi-Layer Perception (MLP) using 2 Hidden Layer for image classification on the CIFAR-10 dataset.
- `Cifar-10_using_3HiddenLayer.ipynb`: A Notebook for training a Multi-Layer Perception (MLP) using 3 Hidden Layer for image classification on the CIFAR-10 dataset.

## Implementation

**MLP with 1 Hidden Layer**: A simple MLP model with a single hidden layer.
- Input: 32x32x3 (CIFAR-10 image size).
- Hidden Layer: 256 units with ReLU Activation.
- Output: 10 classes (CIFAR-10 categories).

```python
model = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(32*32*3, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

**MLP with 2 Hidden Layer**: The second model includes 2 hidden layers.
- Input: 32x32x3 (CIFAR-10 image size).
- Hidden Layers: Two layers with 256 units each, and ReLU activation.
- Output: 10 classes (CIFAR-10 categories).

```python
model = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(32*32*3, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

**MLP with 3 Hidden Layer**: The third model uses 3 hidden layers for increased depth and capacity.
- Input: 32x32x3 (CIFAR-10 image size)
- Hidden Layers: Three layers with 256 units each, and ReLU activation
- Output: 10 classes (CIFAR-10 categories)

```python
model = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(32*32*3, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

- **CNN model**: The CNN model with multiple convolutional layers
- Input: 32x32x3 (CIFAR-10 image size)
- Convolutional Layers: Four layers with increasing channel sizes (32, 64, 128, 256)
- Output: 10 classes (CIFAR-10 categories)

```python
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(8*8*256, 128)
        self.dense2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x
```

## Result 

| Model                     | Test Accuracy |
|---------------------------|---------------|
| MLP using 1 Hidden Layer  |    ~ 53%      |
| MLP using 2 Hidden Layer  |    ~ 53%      | 
| MLP using 3 Hidden Layer  |    ~ 52%      |
| CNN                       |    ~ 70%      |

### Conclusion 1: Comparing MLP Models with Different Numbers of Hidden Layers
- Increasing the number of hidden layers in MLP models (from 1 to 3 or much more) does not necessarily lead to a noticeable improvement in performance. In fact, adding more layers might result in slightly worse performance due to potential overfitting or the model becoming harder to train effectively with limited data.

### Conclusion 2: Comparing MLP Model with CNN Model
- CNNs are much more effective for image classification tasks, such as CIFAR-10, compared to MLP models. The convolutional layers in CNNs allow them to capture spatial features in images, which MLPs, being fully connected, struggle to do. As a result, the CNN model provides much better generalization and higher accuracy.

## Requirement

- You can run code on googlecollab, jupyter or Kaggle notebooks, ... If you don't have a virtual enviroment, manually include the nessesary libraries, like:
numpy
tensorflow
matplotlib
pandas
