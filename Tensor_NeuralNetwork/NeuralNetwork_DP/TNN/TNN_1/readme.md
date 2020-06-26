##  Here is the simple introduction of the code's key part:

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

