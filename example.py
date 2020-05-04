import numpy as np
import TensorAlgebra

I=50
J=50
K=50
F=5
A = np.random.normal(size=[I, F]).astype(
        float)
while np.linalg.matrix_rank(A) != F:
    A = np.random.normal(size=[I, F])

B = np.random.normal(size=[J, F]).astype(
        float)
while np.linalg.matrix_rank(B) != F:
    B = np.random.normal(size=[J, F]).astype(
            float)

C = np.random.normal(size=[K, F]).astype(
        float)
while np.linalg.matrix_rank(C) != F:
    C = np.random.normal(size=[K, F]).astype(
            float)

tensor=TensorAlgebra.kruskal_to_tensor([A,B,C])
[estA,estB,estC],is_converge=TensorAlgebra.parafac(tensor,F,n_iter_max=3000,init='svd',tol=10e-15)
estTensor=TensorAlgebra.kruskal_to_tensor([estA,estB,estC])
loss=TensorAlgebra.norm(tensor-estTensor,order=2)
print("l2 norm of loss is:{}".format(loss))
print(tensor)
print("---------------------------")
print(estTensor)