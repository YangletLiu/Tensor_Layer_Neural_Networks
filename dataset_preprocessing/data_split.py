import cv2
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import os
import errno

import numpy as np
import torch

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(384),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def dct(x, norm=None):
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

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def preprocess_imagenet(img, block_size=(6, 6)):
    img = downsample_img(img, block_size=block_size, total_num_nets=36)
    img = dct(img)
    img = img[:, :, :, 0:2]
    return img


def downsample_img(img, block_size, total_num_nets):
    c_, m_, n_ = img.shape
    row_step, col_step = block_size
    assert total_num_nets == row_step * col_step, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % row_step == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_step == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    components = []
    for row in range(row_step):
        for col in range(col_step):
            components.append(img[:, row::row_step, col::col_step])
    components = torch.stack(components, dim=-1)

    return components


def resize_image(inputFileName, input_str, sub_num = 2):
    try:
        out_path = []
        for i in range(0, sub_num):
            out_path.append(input_str + f"-sub{i}")

            if not os.path.exists(os.path.dirname(out_path[i])):
                try:
                    os.makedirs(os.path.dirname(out_path[i]))
                except OSError as exc:  # Guard against race condition
                    print("OSError ", inputFileName)
                    if exc.errno != errno.EEXIST:
                        raise

        assert out_path != inputFileName
        im = cv2.imread(inputFileName)

        im_resize = preprocess_imagenet(transform(im)).numpy().transpose(1, 2, 0, 3)

        for i in range(0, sub_num):
            cv2.imwrite(out_path[i], im_resize[:, :, :, i])

        print("success!", inputFileName)

    except:
        print("general failure ", inputFileName)


def main():
    path = '/xfs/colab_space/yanglet/imagenet21k' # might need to edit this
    import os
    from glob import glob
    print("scanning files...")
    files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
    print("done, start resizing")

    pool = ThreadPool(8)
    resize_image_fun = partial(resize_image, input_str='imagenet21k')
    pool.map(resize_image_fun, files)


if __name__ == '__main__':
    main()

