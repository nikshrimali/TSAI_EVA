import argparse
from sys import platform

import os
os.chdir(r'D:\Python Projects\EVA\15_TheCapStone\models_all\YoloV3')

from utils.datasets import *
from utils.utils import *


def yolo_detect(model, save_img=False):


    img_size = 512
    out = r'D:\Python Projects\EVA\15_TheCapStone\custom_data\output\Yolo_Output'
    source = r'D:\Python Projects\EVA\15_TheCapStone\custom_data\images_test'
    conf_thres = 0.02
    iou_thres = 0.6
    multi_label=False
    agnostic = False
    names = ['hardhat', 'vest', 'mask', 'boots']
    
    dataset = LoadImages(source, img_size=img_size)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    device = "cuda"


    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Eval mode
    model.to(device).eval()

    # Run inference after normalizing the image
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device))
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred, _ = model(img) # removed augment
        pred = pred[0]
        # print(pred)


        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.6, classes=False, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # print(det)

            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])



            # Save results (image with detections)

            # if dataset.mode == 'images':
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            cv2.imwrite(save_path, im0)
