# Convolutional NeuralNetwork
  We use `tensor convolutional-layer` network to classify the MNIST dataset.

##  NeuralNetwork structure
* Define the neuralnetwork module
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # input channel = 1(grayscalar)
                out_channels=16,    # out channel(self-defined)
                kernel_size=3,      # kernel_size
                stride=1,           # convolutional ste
                padding=1           # padding = (kernel_size - strider) / 2
            ),
            nn.ReLU(),              #activation
            nn.MaxPool2d(kernel_size=2)   #MaxPool
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.pred = nn.Linear(32*7*7,10)    # formulize the output to 10 classes
```

* Forward Process
```python
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x
```

##  Result
### Explanation of Result
As dipicted in the result graph, we can see a clear grow tendency of the `Loss` line with epoch.<br>

Cause the network performances extraordinarily well after the first training epoch(test accuracy up to 98% after the first epoch)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/CNN/cnn_res.png)
