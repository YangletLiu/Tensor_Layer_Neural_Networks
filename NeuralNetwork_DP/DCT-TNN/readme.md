# DCT Tensor NeuralNetwork 

* Key codes: DCT and IDCT transforms

```python
    def dct_tensor_product(tensorA, tensorB):
        a_l, a_m, a_n = tensorA.shape
        b_l, b_m, b_n = tensorB.shape
        dct_a = torch.transpose(dct(torch.transpose(tensorA, 0, 2)), 0, 2)      #swap the '0' axis and '2' axis because of the specific transfer axis 
        dct_b = torch.transpose(dct(torch.transpose(tensorB, 0, 2)), 0, 2)

        dct_product = []
        for i in range(a_l):
            dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))        #do the frontal slices matrix multiplication in DCT domain
        dct_products = torch.stack(dct_product)

        idct_product = torch.transpose(idct(torch.transpose(dct_products, 0, 2)), 0, 2).reshape(a_l, a_m, b_n)   #swap the axis and do the inverse DCT

        return idct_product
```
