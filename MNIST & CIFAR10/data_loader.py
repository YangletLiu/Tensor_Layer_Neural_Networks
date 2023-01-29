from torchvision import transforms, datasets

def get_dataset(dataset):
    trainset = None
    testset = None

    if dataset == "MNIST":
        transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(root="../datasets", train=True, download=True, transform=transform_MNIST)
        testset = datasets.MNIST(root="../datasets", train=False, download=True, transform=transform_MNIST)

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

        trainset = datasets.CIFAR10(root="../datasets", train=True, transform=transform_train, download=True)
        testset = datasets.CIFAR10(root="../datasets", train=False, transform=transform_test, download=True)

    return trainset, testset