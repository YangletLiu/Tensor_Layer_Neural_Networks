# Autoencoder

  An autoencoder with: a 4-layer encoder, and correpondinlgy a 4-layer decoder.

##  Network structure
* define the encoder and decoder layers:<br>
  Initial dimensions:`28*28` and the follwing layers' dimensions are `128`,`64`,`12`,`3`
```python
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())
```
* Forward Process(encoder and decoder)
```python
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

##  Result
![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/Autoencoder/autoencoder_res.png)
