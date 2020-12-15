## Main Code File directory 
* new_tnn.py

  test the mnist dataset

* new_tnn_cifar.py and new_tnn_cifar_3channel.py

  test the cifar10 dataset, new_tnn_cifar.py test the one channel. new_tnn_cifar_3channel.py test the RGB channels and separate training.


## Trainning Process
```python 
# begain train
for epoch in range(epochs_num):
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    module.train()

    pbar_train = pkbar.Pbar(name='Epoch '+str(epoch+1)+' training:', target=50000/batch_size)
    for i, data in enumerate(train_loader):
        img, label = data
        imgR = cifar_img_processRGB(img, 0)
        imgR = raw_img(imgR, batch_size, n=32)
        imgR = imgR.to(device)

        imgG = cifar_img_processRGB(img, 1)
        imgG = raw_img(imgG, batch_size, n=32)
        imgG = imgG.to(device)

        imgB = cifar_img_processRGB(img, 2)
        imgB = raw_img(imgB, batch_size, n=32)
        imgB = imgB.to(device)
        label = label.to(device)

        # forward
        outR = module(imgR)
        outG = module(imgG)
        outB = module(imgB)

        # softmax function
        out = (torch.transpose(scalar_tubal_func(outR), 0, 1)+torch.transpose(scalar_tubal_func(outG), 0, 1)+torch.transpose(scalar_tubal_func(outB), 0, 1))/3
        loss = Loss_function(out, label)
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar_train.update(i)
```
## Running Tips
run `python xxxxx.py` to run the code and get the results.
