import cv2
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from flow_warp import forward_warp, backward_warp


if __name__ == "__main__":

    im0 = cv2.imread("im0.png")[np.newaxis, :, :, :]
    im1 = cv2.imread("im1.png")[np.newaxis, :, :, :]
    with open("flow.pkl", "rb+") as f:
        flow = pickle.load(f)
    im0 = torch.FloatTensor(im0).permute(0, 3, 1, 2).contiguous().cuda().requires_grad_(True)
    im1 = torch.FloatTensor(im1).permute(0, 3, 1, 2).contiguous().cuda().requires_grad_(True)
    flow = torch.FloatTensor(flow).contiguous().cuda().requires_grad_(True)
    # print(flow.shape, im0.shape)

    im0_warp = forward_warp(im0, flow)
    mask = torch.ones_like(im0)
    mask_warp = forward_warp(mask, flow)
    im0_warp = im0_warp / mask_warp
    # print(im0_warp.shape)
    cv2.imwrite("im0_warp.png", im0_warp[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    im1_warp = backward_warp(im1, flow)
    cv2.imwrite("im1_warp.png", im1_warp[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))

    loss = F.mse_loss(im0_warp, im1)
    loss.backward()

    # loss = F.mse_loss(im1_warp, im0)
    # loss.backward()