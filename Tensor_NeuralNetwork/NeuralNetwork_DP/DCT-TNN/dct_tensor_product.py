import torch
import numpy as np


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

    return torch.mm(mat_a, mat_b)


a = torch.arange(1, 9, dtype=torch.float).reshape(2, 2, 2)
b = torch.arange(2, 10, dtype=torch.float).reshape(2, 2, 2)
print(dct_mat_product(a, b))
