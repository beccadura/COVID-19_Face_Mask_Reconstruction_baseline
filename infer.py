import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
from PIL import Image
import cv2
from models import UNetSemantic, UNetSemantic2
import argparse
import glob
from losses import *
from metrics import *
import numpy as np

class Predictor():
    def __init__(self):
        self.device = torch.device('cpu')
        self.masking = UNetSemantic().to(self.device)
        self.masking.load_state_dict(torch.load('weights/model_segm_25_100.pth', map_location='cpu'))

        self.inpaint = UNetSemantic2().to(self.device)
        self.inpaint.load_state_dict(torch.load('weights/model_24_124.pth', map_location='cpu')['G'])
        self.inpaint.eval()
        self.criterion_ssim = SSIM(window_size = 11)
        self.criterion_dice = DiceLoss()
        self.img_size = 512

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow = nrow)
        print(f"Save image to {save_img_path}")

    def predict(self, outpath='sample/results.png'):

        SSIM = []
        DICE = []
        SSIM_mask = []
        DICE_mask = []

        jpgFilenamesList = glob.glob('datasets/Dataset1-David-baseline/test/image/*.jpg')
        for image in jpgFilenamesList:
            image = image[-13:-4]
            outpath=f'results/results_{image}.png'
            image2 = 'datasets/Dataset1-David-baseline/test/image_masked/'+image
            img = cv2.imread(image2+'_masked.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            img = img.unsqueeze(0)

            image3 = 'datasets/Dataset1-David-baseline/test/image/'+image
            img_ori = cv2.imread(image3+'.jpg')
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_ori = cv2.resize(img_ori, (self.img_size, self.img_size))
            img_ori = torch.from_numpy(img_ori.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            img_ori = img_ori.unsqueeze(0)
            
            with torch.no_grad():
                mask_outputs = self.masking(img)

                outputs = mask_outputs

                for idx,i in enumerate(outputs):
                    for idx2,i2 in enumerate(i):
                        for idx3,i3 in enumerate(i2):
                            for idx4,i4 in enumerate(i3.data):
                                if i4 >= 0.5:
                                    outputs[idx][idx2][idx3][idx4].data = torch.tensor(1)
                                else:
                                    outputs[idx][idx2][idx3][idx4].data = torch.tensor(0)

                kernel = np.ones((3,3),np.uint8)
                erosion = cv2.erode(np.float32(outputs[0][0].data),kernel,iterations = 1)
                dilation = cv2.dilate(erosion,kernel,iterations = 1)
                dilation = torch.from_numpy(dilation).contiguous()
                dilation = dilation.unsqueeze(0)
                dilation = dilation.unsqueeze(0)

                out = self.inpaint(img, dilation)
                inpaint = img * (1 - dilation) + out * dilation
            masks = img * (1 - dilation) + dilation

            loss_ssim = self.criterion_ssim(inpaint, img_ori)
            print(loss_ssim)
            SSIM.append(loss_ssim)
            loss_dice = 1 - self.criterion_dice(inpaint, img_ori)
            print(loss_dice)
            DICE.append(loss_dice)

            loss_ssim_mask = self.criterion_ssim(out * dilation, img_ori * dilation)
            print(loss_ssim_mask)
            SSIM_mask.append(loss_ssim_mask)
            loss_dice_mask = 1 - self.criterion_dice(out * dilation, img_ori * dilation)
            print(loss_dice_mask)
            DICE_mask.append(loss_dice_mask)
            
            self.save_image([img, masks, inpaint, img_ori], outpath, nrow=4)

        print("DICE Loss: ", np.mean(DICE))
        print("SSIM Loss: ", np.mean(SSIM))
        print("DICE Mask Loss: ", np.mean(DICE_mask))
        print("SSIM Mask Loss: ", np.mean(SSIM_mask))

        


if __name__ == '__main__':
    model = Predictor()
    model.predict()