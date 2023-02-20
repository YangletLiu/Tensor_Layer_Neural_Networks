from torchvision import transforms, datasets
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import torch

def get_dataset(dataset):
    trainset = None
    testset = None

    if dataset == "MNIST":
        transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        # trainset = datasets.MNIST(root="D:/考研/复试/导师/资料/本周在看文献/张量/tensorForDNN/实验/Tensor_Layer_Neural_Networks-master/datasets", train=True, download=True, transform=transform_MNIST)
        # testset = datasets.MNIST(root="D:/考研/复试/导师/资料/本周在看文献/张量/tensorForDNN/实验/Tensor_Layer_Neural_Networks-master/datasets", train=False, download=True, transform=transform_MNIST)
        trainset = datasets.MNIST(root="/xfs/home/tensor_zy/zhangjie/datasets", train=True, download=True, transform=transform_MNIST)
        testset = datasets.MNIST(root="/xfs/home/tensor_zy/zhangjie/datasets", train=False, download=True, transform=transform_MNIST)

    if dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010])
        ])

        trainset = datasets.CIFAR10(root="D:\\考研\\复试\\导师\\资料\\本周在看文献\\张量\\tensorForDNN\\实验\\Tensor_Layer_Neural_Networks-master\\datasets", train=True, transform=transform_train, download=True)
        testset = datasets.CIFAR10(root="D:\\考研\\复试\\导师\\资料\\本周在看文献\\张量\\tensorForDNN\\实验\\Tensor_Layer_Neural_Networks-master\\datasets", train=False, transform=transform_test, download=True)

    return trainset, testset

class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        # mean=(0.485, 0.456, 0.406),
        # std=(0.229, 0.224, 0.225),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)