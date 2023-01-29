import numpy as np
import torch
from torch import nn


def low_rank_matrix_decompose_FC_layer(layer, r):
    print("'low rank matrix' decompose one FC layer")
    # y = xW + b ==>  y = xDC + b
    lj, lj_1 = layer.weight.data.shape

    D = nn.Linear(in_features=lj_1,
                   out_features=r,
                   bias=False)
    C = nn.Linear(in_features=r,
                   out_features=lj,
                   bias=True)

    C.weight.data = torch.randn(lj, r) * torch.sqrt(torch.tensor(2 / (lj + r)))
    D.weight.data = torch.randn(r, lj_1) * torch.sqrt(torch.tensor(2 / (r + lj_1)))
    C.bias.data = torch.randn(lj, ) * torch.sqrt(torch.tensor(2 / (lj + 1)))

    new_layers = [D, C]
    return nn.Sequential(*new_layers)

def low_rank_matrix_decompose_nested_FC_layer(layer, rank=10):
    modules = layer._modules
    for key in modules.keys():
        l = modules[key]
        if isinstance(l, nn.Sequential):
            modules[key] = low_rank_matrix_decompose_nested_FC_layer(l)
        elif isinstance(l, nn.Linear):
            fc_layer = l
            sp = fc_layer.weight.data.numpy().shape
            if rank >= min(sp):
                continue
            modules[key] = low_rank_matrix_decompose_FC_layer(fc_layer, rank)
    return layer


# decomposition
def decompose_FC(model, mode, rank=10):
    model.eval()
    model.cpu()
    layers = model._modules
    for i, key in enumerate(layers.keys()):
        if isinstance(layers[key], torch.nn.modules.Linear):
            fc_layer = layers[key]
            sp = fc_layer.weight.data.numpy().shape
            if rank >= min(sp):
                continue
            if mode == "low_rank_matrix":
                layers[key] = low_rank_matrix_decompose_FC_layer(fc_layer, rank)
        elif isinstance(layers[key], nn.Sequential):
            if mode == "low_rank_matrix":
                layers[key] = low_rank_matrix_decompose_nested_FC_layer(layers[key])
    return model

def dct(x, norm=None, device="cpu"):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    if torch.__version__ > "1.7.1":
        Vc = torch.view_as_real(torch.fft.fft(v))
    else:
        Vc = torch.rfft(v, 1, onesided=False)

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None, device="cpu"):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (torch.arange(x_shape[-1], dtype=X.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    if torch.__version__ > "1.7.1":
        v = torch.fft.ifft(torch.view_as_complex(V)).real
    else:
        v = torch.irfft(V, 1, onesided=False)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


# circulant convolution part
def torch_tensor_Bcirc(tensor, l, m, n):
    bcirc_A = []
    for i in range(l):
        bcirc_A.append(torch.roll(tensor, shifts=i, dims=0))
    return torch.cat(bcirc_A, dim=2).reshape(l * m, l * n)


def dct_tensor_product(tensorA, tensorB):
    a_l, a_m, a_n = tensorA.shape
    b_l, b_m, b_n = tensorB.shape
    dct_a = torch.transpose(dct(torch.transpose(tensorA, 0, 2)), 0, 2)
    # print(dct_a)
    dct_b = torch.transpose(dct(torch.transpose(tensorB, 0, 2)), 0, 2)
    # print(dct_b)

    dct_product = []
    for i in range(a_l):
        dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))
    dct_products = torch.stack(dct_product)

    idct_product = torch.transpose(idct(torch.transpose(dct_products, 0, 2)), 0, 2).reshape(a_l, a_m, b_n)

    return idct_product


# Loss function(scalar tubal softmax function)
def h_func_dct(lateral_slice):
    l, m, n = lateral_slice.shape

    dct_slice = dct(lateral_slice)

    tubes = [dct_slice[i, :, 0] for i in range(l)]

    # todo: parallelism here, use tensor's batch operation
    h_tubes = []
    for tube in tubes:
        tube_sum = torch.sum(torch.exp(tube))
        h_tubes.append(torch.exp(tube) / tube_sum)
    #######################################################

    res_slice = torch.stack(h_tubes, dim=0).reshape(l, m, n)

    idct_a = idct(res_slice)

    return torch.sum(idct_a, dim=0)


def scalar_tubal_func(output_tensor):
    l, m, n = output_tensor.shape

    lateral_slices = [output_tensor[:, :, i].reshape(l, m, 1) for i in range(n)]

    h_slice = []
    for slice in lateral_slices:
        h_slice.append(h_func_dct(slice))

    pro_matrix = torch.stack(h_slice, dim=2)
    return pro_matrix.reshape(m, n)


# process raw
def raw_img(img, seg_length=0):
    """
        :param img: (batch_size, channel=1, n1, n2)
        :return (n2, n1, batch_size)
    """
    img = torch.squeeze(img)
    if seg_length:
        sp = img.shape
        img = img.reshape(sp[0], -1, seg_length)
    ultra_img = img.permute([2, 1, 0])
    return ultra_img

def downsample_img(img, block_size, num_nets):
    batch_, c_, m_, n_ = img.shape
    row_step, col_step = block_size
    row_blocks = m_ // row_step
    col_blocks = n_ // col_step
    assert num_nets == row_blocks * col_blocks, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % row_step == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_step == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    components = []
    for row in range(row_step):
        for col in range(col_step):
            components.append(img[:, :, row::row_step, col::col_step].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(row_step * col_step):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))

    return img

def block_img(img, block_size, num_nets):
    batch_, c_, m_, n_ = img.shape
    block_row, block_col = block_size
    row_blocks = m_ // block_row
    col_blocks = n_ // block_col
    assert num_nets == row_blocks * col_blocks, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % block_row == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % block_col == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    components = []
    for row_block_idx in range(row_blocks):
        for col_block_idx in range(col_blocks):
            components.append(img[:,
                                  :,
                                  row_block_idx * block_row : row_block_idx * block_row + block_row,
                                  col_block_idx * block_col : col_block_idx * block_col + block_col].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(row_blocks * col_blocks):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))
    # print(img.shape)
    # exit(0)

    return img


def preprocess_mnist(img, block_size,method, num_nets, trans, device):
    # mi, ma = -0.4242, 2.8215
    # img += (torch.rand_like(img, device=device) * (ma - mi) - mi)
    if method == "downsample":
        img = downsample_img(img, block_size=block_size, num_nets=num_nets)
    elif method == "block":
        img = block_img(img, block_size=block_size, num_nets=num_nets)

    if trans:
        img = dct(img, device=device)
    return img

def fusing(num_nets, p, train_loss):
    p = 0.3
    rank_list = np.argsort(train_loss)
    fusing_weight = [0] * num_nets
    for i in range(num_nets):
        fusing_weight[rank_list[i]] = p * np.power((1 - p), i)

    return fusing_weight