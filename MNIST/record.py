import csv

import numpy as np
from matplotlib import pyplot as plt


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, model_name):
    # write csv
    with open(model_name+"_mnist_testloss.csv", "w", newline='', encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["Test Loss:"])
        f_csv.writerows(enumerate(test_loss,1))
        f_csv.writerow(["Train Loss:"])
        f_csv.writerows(enumerate(train_loss,1))
        f_csv.writerow(["Test Acc:"])
        f_csv.writerows(enumerate(test_acc,1))
        f_csv.writerow(["Train Acc:"])
        f_csv.writerows(enumerate(train_acc,1))

    # draw picture
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title(model_name + " Loss on MNIST ")
    plt.plot(np.arange(len(test_loss)), test_loss, color="red", label="TestLoss",linestyle="-")
    plt.plot(np.arange(len(train_loss)), train_loss, color="blue", label="TrainLoss",linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title(model_name + " Accuracy on MNIST ")
    plt.plot(np.arange(len(test_acc)), test_acc, color="green", label="TestAcc",linestyle="-")
    plt.plot(np.arange(len(train_acc)), train_acc, color="orange", label="TrainAcc",linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")

    plt.legend()
    plt.show()

    plt.savefig("./"+model_name+"_mnist.jpg")