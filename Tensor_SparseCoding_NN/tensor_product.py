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
def back_propagation(delta_A_j_1,W_j,A_j):
    Z_j_1 = t_product(W_j, A_j) + B_j
    A_j_t = A_j.swapaxes(1,2)
    delta_W_j = t_product((delta_A_j_1*Z_j_1),A_j_t)
    delta_B_j = A_j.sum(2).reshape(2,2,1)
    return delta_W_j,delta_B_j

#back-propagation example
delta_A_j_1 = np.array(range(2,14)).reshape(2,2,3)
W_j = np.array(range(1,9)).reshape(2,2,2)
A_j = np.array(range(1,13)).reshape(2,2,3)
B_j = np.array(range(1,5)).reshape(2,2,1)
# print(back_propagation(delta_A_j_1,W_j,A_j))

#frequency domain t-product example
pass


#f_scalar_tube-wise function,which is implemented on every lateral slice of the tensor
def f_tube_wise(lateral_slice):
    #later slices
    # X_i = np.array(range(1,10)).reshape(3,3,1)
    l,m,n = lateral_slice.shape
    #tubes
    tubes =  [lateral_slice[:,i,0].reshape(l,1,1) for i in range(lateral_slice.shape[1])]
    tubes_sum = np.zeros(l).reshape(l,1,1)
    for i in tubes:
        # print(np.exp(i))
        tubes_sum += np.exp(i)
    tubes_sum = 1/tubes_sum
    # print(tubes_sum)
    h_tubes = [t_product(tubes_sum,i) for i in tubes]
    g_tubes = [i.sum() for i in h_tubes]
    # print(g_tubes)
    return g_tubes

#scalar_tubal_softmax function ,
#which is implemented on 3D-tensor to get the matrix of probabilities
def F_scalar_tubal_softmax(A_N):
    l,m,n = A_N.shape
    lateral_slices = [A_N[:,:,i] for i in range(A_N.shape[2])]
    X = []
    for tube in lateral_slices:
        tube = tube.reshape(l,m,1)
        res_tube = f_tube_wise(tube)
        # print(res_tube)
        X.append(res_tube)
    return np.vstack(X).transpose()

A_N = np.array(range(1,9)).reshape(2,2,2)
print(F_scalar_tubal_softmax(A_N))