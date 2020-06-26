##  The key parts of the code done by us:

* DCT and IDCT transforms:
```python
def dct(x, norm=None)

def idct(x, norm=None)
```
* Tensor product by circulant convolution in time domain
```python
def torch_tensor_Bcirc(tensor, l, m, n)

def torch_tensor_product(tensorA, tensorB)
```
* Scalar tubal softmax function implemented on the last output layer
```python
def h_func_dct(lateral_slice)

def scalar_tubal_func(output_tensor)
```
##  Here is the neuralnetwork structure:

* Define customized parameters which needs to be optimized in training process
```python
    def __init__(self):
        super(tNN, self).__init__()
        """
        use the nn.Parameter() and 'requires_grad = True' 
        to customize parameters which are needed to optimize
        """
        self.W_1 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
        self.B_1 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
        self.W_2 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
        self.B_2 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
        self.W_3 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
        self.B_3 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
        self.W_4 = nn.Parameter(torch.randn((28, 10, 28), requires_grad=True, dtype=torch.float))
        self.B_4 = nn.Parameter(torch.randn((28, 10, 1), requires_grad=True, dtype=torch.float))
```

* Forward algorithms using tensor-product(time-domain)
```python
    def forward(self, x):
        """
        torch_tensor_product is redefined by torch to complete the tensor-product process
        :param x: x is the input 3D-tensor with shape(l,m,n)
                     'n' denotes the batch_size
        :return: this demo defines an one-layer networks,
                    whose output is processed by one-time tensor-product and activation
        """
        x = torch_tensor_product(self.W_1, x) + self.B_1
        x = F.relu(x)
        x = torch_tensor_product(self.W_2, x) + self.B_2
        x = F.relu(x)
        x = torch_tensor_product(self.W_3, x) + self.B_3
        x = F.relu(x)
        x = torch_tensor_product(self.W_4, x) + self.B_4
        x = F.relu(x)
        return x
```

* Loss function:CrossEntropy & Optimizer: SGD
```python
Loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(module.parameters(), lr=lr_rate)
```
##  Training Process
```python
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = raw_img(img, img.size(0), 28)   #formulize the input picture into 3D-tensor
        
        #if use gpu to accelerate the training
        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # forward
        out = module(img) / 1e5         #the output tensor includes large numbers,which can easily cause overflow error in the following softmax function
                                        #because we use the exponential operation in scalar tubal softmax function

        # softmax function
        out = torch.transpose(scalar_tubal_func(out), 0, 1)
        
        #loss function
        loss = Loss_function(out, label)
        running_loss += loss.item()
        
        #test training accuracy
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        #optimize parameters
        optimizer.step()
```

##  Running Tips
You can first set the three main super parameters: batch_size,learning_rate and epoch_numbers :
```python 
batch_size = 100
lr_rate = 0.1
epochs_num = 100
```
Then run `python tnn-4.py` to run the code and get the results.
