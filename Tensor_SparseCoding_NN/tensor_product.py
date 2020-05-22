import numpy as np
import PIL
import matplotlib.pyplot as plt

#circulant tensor.l,m,n denote the shape of tensor
def tensor_Bcirc(tensor,l,m,n):
    #frontal_slices
    frontal_slices = [tensor[i,:,:] for i in range(l)]
    frontal_slices.reverse()

    #circ
    circ_slices = []
    for i in range(l):
        frontal_slices.insert(0,frontal_slices.pop())
        for j in frontal_slices:
            circ_slices.append(j)

    circ = []
    for i in range(l):
        circ.append(np.hstack(circ_slices[l*i:l*i+l]))
    ulti_circ = np.vstack(circ).reshape(l*m,l*n)

    return ulti_circ

#tensor-product
def t_product(tensorA,tensorB):
    a_l,a_m,a_n = tensorA.shape
    b_l,b_n,b_p = tensorB.shape

    if a_l==b_l and a_n == b_n :
        circA = tensor_Bcirc(tensorA,a_l,a_m,a_n)
        circB = tensor_Bcirc(tensorB,b_l,b_n,b_p)
        return circA.dot(circB[:,0:b_p]).reshape(a_l,a_m,b_p)
    else:
        print('Shape Error')


#back-propagation,we assume the front layer's loss were known(delta_A_j_1)
def back_propagation(delta_A_j_1,W_j,A_j,B_j):
    Z_j_1 = t_product(W_j, A_j) + B_j
    A_j_t = A_j.swapaxes(1,2)
    delta_W_j = t_product((delta_A_j_1*Z_j_1),A_j_t)
    delta_B_j = A_j.sum(2).reshape(2,2,1)
    return delta_W_j,delta_B_j

#tensor-product example
def t_product_example():
    W_j = np.array(range(1,9)).reshape(2,2,2)
    A_j = np.array(range(1,13)).reshape(2,2,3)
    print(t_product(W_j,A_j))
