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


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc, num_nets, fusing_plan, fusing_num, filename):
    # write csv
    with open(filename+'_testloss.csv', 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)

        f_csv.writerow(["Test Acc:"])
        for idx in range(len(test_acc)):
            f_csv.writerow([idx + 1] + test_acc[idx])

        f_csv.writerow(["Test Loss:"])
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + test_loss[idx])

        f_csv.writerow(["Fusing Test Acc:"] + fusing_plan)
        for idx in range(len(fusing_test_acc)):
            f_csv.writerow([idx + 1] + fusing_test_acc[idx])

        f_csv.writerow(["Fusing Test Loss:"] + fusing_plan)
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + fusing_test_loss[idx])

        f_csv.writerow(["Train Acc"])
        for idx in range(len(train_acc)):
            f_csv.writerow([idx + 1] + train_acc[idx])

        f_csv.writerow(["Train Loss"])
        for idx in range(len(train_loss)):
            f_csv.writerow([idx + 1] + train_loss[idx])

    # draw picture
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)
    fusing_test_acc = np.array(fusing_test_acc)
    fusing_test_loss = np.array(fusing_test_loss)
    train_acc = np.array(train_acc)
    train_loss = np.array(train_loss)

    plt.cla()
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title('block-fc-8L-subnets-28 Loss on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_loss[:, i])), test_loss[:, i], label='TestLoss_{}'.format(i + 1), linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_loss[:, i])), fusing_test_loss[:, i],
                 label='FusingTestLoss_{}'.format(i + 1), linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_loss[:, i])), train_loss[:, i], label='TrainLoss_{}'.format(i + 1), linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('block-fc-8L-subnets-28 Accuracy on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_acc[:, i])), test_acc[:, i], label='TestAcc_{}'.format(i + 1), linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_acc[:, i])), fusing_test_acc[:, i], label='FusingTestAcc_{}'.format(i + 1),
                 linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_acc[:, i])), train_acc[:, i], label='TrainAcc_{}'.format(i + 1), linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./block_fc_8L_subnets_28_mnist_no_noise.jpg')