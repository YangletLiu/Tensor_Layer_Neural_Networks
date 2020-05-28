from tensor_product import t_product
import numpy as np

def fft_product(tensorA,tensorB):
    a_l,a_m,a_n = tensorA.shape
    b_l,b_m,b_n = tensorB.shape

    #FFT along the third dimension
    fft_A = np.fft.fft(tensorA,n=None,axis=0)
    fft_B = np.fft.fft(tensorB,n=None,axis=0)

    #matrix product
    fft_products = []
    for n in range(a_l):
        fft_products.append(np.dot(fft_A[n,:,:],fft_B[n,:,:]))
    fft_product = np.vstack(fft_products).reshape(a_l,a_m,b_n)

    #inverse FFT
    ifft_product = np.fft.ifft(fft_product,n=None,axis=0)

    return np.real(ifft_product)

#example of verification of FFT_tensorProduct(frequency domain) & t_product(time domain)
a = np.array(range(1,9)).reshape(2,2,2)
b = np.array(range(2,10)).reshape(2,2,2)

#time domain tensor product
tensor_product1 = t_product(a,b)
print("tensor product in time domain:")
print(tensor_product1)
print("#"*100)

#frequency domain tensor product by FFT
tensor_product2 = fft_product(a,b)
print("tensor product in frequency domain:")
print(tensor_product2)