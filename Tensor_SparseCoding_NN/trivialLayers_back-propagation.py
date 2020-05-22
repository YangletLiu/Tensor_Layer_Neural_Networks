import numpy as np
from tensor_product import t_product

#back-propagation,we assume the front layer's loss were known(delta_A_j_1)
def back_propagation(delta_A_j_1,W_j,A_j,B_j):
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
print(back_propagation(delta_A_j_1,W_j,A_j,B_j))