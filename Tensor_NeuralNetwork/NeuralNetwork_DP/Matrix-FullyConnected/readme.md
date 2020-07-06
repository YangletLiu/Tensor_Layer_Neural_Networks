# Matrix-FullyConnected NeuralNetwork
  In this part, we create a `4-layer linear fully-connected network`(300,200,100,10)

##  Network structure
* Define the neuralnetwork module
```python
class neuralNetwork(nn.Module):
    
    """in_dem represents the dimension of input
        n_hidden_1,n_hidden_2,n_hidden_3 denotes the three hidden layers' number"""

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(neuralNetwork, self).__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, out_dim),
            nn.ReLU(True)
        )
```

* Forward Process
```python
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
```

##  Result


* 4-layer matrix neuralnetwork (2020.7.6)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/Matrix-FullyConnected/mnn4_Loss.png)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/Matrix-FullyConnected/mnn4_Acc.png)

* 8-layer matrix neuralnetwork (2020.7.6)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/Matrix-FullyConnected/mnn8_Loss.png)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/Matrix-FullyConnected/mnn8_Acc.png)
