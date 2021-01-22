# dct in the beginning and idct at the end

import torch
import numpy as np
import torch_dct.dct as dct


def toeplitz_mat(tensor):
    l, m, n = tensor.shape

    frontal_slices = [tensor[i, :, :] for i in range(l)]

    circ_slices = []
    for i in range(l):
        for j in range(l):
            circ_slices.append(frontal_slices[np.abs(i - j)])

    circ = []
    for i in range(l):
        circ.append(torch.cat(circ_slices[l * i:l * i + l], dim=1))
    ulti_circ = torch.cat(circ).reshape(l * m, l * n)
    return ulti_circ


def hankel_matrix(tensor):
    l, m, n = tensor.shape

    frontal_slices = []
    for i in range(1, l):
        frontal_slices.append(tensor[i, :, :].flip([0]))
    frontal_slices.append(torch.zeros(2, 2))
    frontal_slices.reverse()

    c = toeplitz_mat(torch.stack(frontal_slices))

    return torch.flip(c, dims=[0])


def dct_mat_product(tensorA, tensorB):
    mat_a = toeplitz_mat(tensorA) + hankel_matrix(tensorA)
    mat_b = toeplitz_mat(tensorB) + hankel_matrix(tensorB)

    mat_b = mat_b[:, 0:tensorB.shape[2]]

    return torch.mm(mat_a, mat_b).reshape(2,2,2)


def dct_test1():
    dct_a = torch.transpose(dct.dct(torch.transpose(a, 0, 2)), 0, 2)
    # print(dct_a)
    dct_b = torch.transpose(dct.dct(torch.transpose(b, 0, 2)), 0, 2)
    # print(dct_b)

    dct_product = []
    for i in range(2):
        dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))
    dct_products = torch.stack(dct_product)

    idct_product = torch.transpose(dct.idct(torch.transpose(dct_products, 0, 2)), 0, 2).reshape(2,2,2)

    print(idct_product)

def dct_test2():
    lateral_slices_a = torch.split(a, split_size_or_sections=1, dim=2)
    dct_lateral_slices_a = [dct.dct(i) for i in lateral_slices_a]
    dct_a = torch.cat(dct_lateral_slices_a, dim=2)
    # print(dct_a)

    lateral_slices_b = torch.split(b, split_size_or_sections=1, dim=2)
    dct_lateral_slices_b = [dct.dct(i) for i in lateral_slices_b]
    dct_b = torch.cat(dct_lateral_slices_b, dim=2)
    # print(dct_b)

    dct_product = []
    for i in range(2):
        dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))
    dct_products = torch.stack(dct_product)
    # print(dct_products)

    lateral_slices_product = torch.split(dct_products, split_size_or_sections=1, dim=2)
    idct_lateral_slices_product = [dct.idct(i) for i in lateral_slices_product]
    idct_product = torch.cat(idct_lateral_slices_product, dim=2)

    print(idct_product)

def dct_test3():
    frontal_slices_a = torch.split(a, split_size_or_sections=1, dim=0)
    # print(frontal_slices_a)
    dct_frontal_slices_a = [dct.dct(i) for i in frontal_slices_a]
    dct_a = torch.stack(dct_frontal_slices_a).reshape(2, 2, 2)
    # print(dct_a)

    frontal_slices_b = torch.split(b, split_size_or_sections=1, dim=0)
    # print(frontal_slices_b)
    dct_frontal_slices_b = [dct.dct(i) for i in frontal_slices_b]
    dct_b = torch.stack(dct_frontal_slices_b).reshape(2, 2, 2)
    # print(dct_b)

    dct_product = []
    for i in range(2):
        dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))
    dct_products = torch.stack(dct_product)

    frontal_slices_product = torch.split(dct_products, split_size_or_sections=1, dim=0)
    # print(frontal_slices_product)
    idct_frontal_slices_product = [dct.idct(i) for i in frontal_slices_product]
    idct_product = torch.stack(idct_frontal_slices_product)

    print(idct_product)

a = torch.arange(1, 9, dtype=torch.float).reshape(2, 2, 2)
b = torch.arange(2, 10, dtype=torch.float).reshape(2, 2, 2)
print('Expected result:')
print(dct_mat_product(a, b))
print("#"*100)

print('Test-1 result:')
dct_test1()
print("#"*100)
print('Test-2 result:')
dct_test2()
print("#"*100)
print('Test-3 result:')
dct_test3()
print("#"*100)

