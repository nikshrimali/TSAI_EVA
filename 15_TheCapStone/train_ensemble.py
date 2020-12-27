import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
os.chdir(r'D:\Python Projects\EVA\15_TheCapStone\models_all\YoloV3')

from test_ensemble import test_model  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *


results_file = 'results.txt'

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train(model):
    # Value setting for hyperparameters
    device = "cuda"
    data = r'D:\Python Projects\EVA\15_TheCapStone\custom_data\custom.data' # Path in which train images are located # path = './cfg/yolov3-custom.cfg'
    epochs = 5  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = 4
    accumulate = 4  # effective bs = batch_size * accumulate = 16 * 4 = 64
    img_size = 512
    imgsz_test = 512
   
    # Configure run
    init_seeds() # Set seeds
    data_dict = parse_data_cfg(data) #parse the config
    print(data_dict['valid'])
    train_path = r"D:\Python Projects\EVA\15_TheCapStone\custom_data\train.txt"
    test_path = r"D:\Python Projects\EVA\15_TheCapStone\custom_data\test.txt"
    # custom_names = r".\data\customdata\custom.names"
    nc = 4  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    cfg = './cfg/yolov3-custom.cfg'

    # Initialize model
    # model = model

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else
    
    # Initializing the optimizer
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0

    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    
    # train_path = 
    # LR scheduler
#################################################################################################
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Reading custom dataset and applying custom augmentation
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=False,
                                  hyp=hyp,  # augmentation hyperparameters
                                  )
############################################################################################
    # Dataloader - Loading the dataset
    batch_size = min(batch_size, len(dataset))
    nw = 1 #min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=True,  # Shuffle=True unless rectangular training isused
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,hyp=hyp),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)


    # Model Training parameters
    model.nc = 4  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights konse wale

    # Model EMA - Is this necessary? EMA - Moving averages weight setting
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    # print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, ))
    
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    # Training loop
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # loss
        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64


            loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # Plot images with bounding boxes
            # print('@@@@@ ni values are',ni)
            if ni < 1:
                f = 'train_batch%g.png' % i  # filename
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)      

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if final_epoch:  # Calculate mAP
            results, maps = test_model(cfg,
                                data,
                                batch_size=batch_size,
                                img_size=imgsz_test,
                                model=ema.ema,
                                save_json=final_epoch,
                                single_cls=False,
                                dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
               # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
        # end epoch ----------------------------------------------------------------------------------------------------
