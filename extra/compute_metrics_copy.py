import os, math
import numpy as np
import scipy.signal
from typing import List, Optional
from PIL import Image
import os
import torch
import configargparse

__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)] == target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

# create a function (this my favorite choice)
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == '__main__':

    # parser = configargparse.ArgumentParser()
    # parser.add_argument("--exp", type=str, help="folder of exps")
    # parser.add_argument("--paramStr", type=str, help="str of params")
    # args = parser.parse_args()

    # datanames = ['drums','hotdog','materials','ficus','lego','mic','ship','chair'] #['ship']#
    # gtFolder = "/root/qing/StyleRF/data/nerf_synthetic"
    # expFolder = "/root/qing/StyleRF/log/nerf"

    # datanames = ['room','fortress', 'flower','orchids','leaves','horns','trex','fern'] #['ship']#
    datanames = ['flower']
    gtFolder = "/root/autodl-tmp/StyleRF/data/nerf_llff_data/"
    expFolder = "/root/autodl-tmp/StyleRF/log_style/flower/flower/imgs_path_all/000d655562800587aceb35c35ed4c47cc"
    # paramStr = args.paramStr
    fileNum = 10

    # expitems = os.listdir(expFolder)
    finalFolder = "/root/autodl-tmp/StyleRF/log_style/finals"
    outFile = f'{finalFolder}/metrics.txt'
    os.makedirs(finalFolder, exist_ok=True)

    # expitems.sort(reverse=True)
    criterion = RMSELoss

    with open(outFile, 'w') as f:
        all_psnr = []
        all_ssim = []
        all_alex = []
        all_vgg = []
        all_rmse = []
        for dataname in datanames:

            # gtstr = gtFolder + "/" + dataname + "/test/r_%d.png"
            # expname = findItem(expitems, f'{dataname}')
            # print("expname: ", expname)
            # if expname is None:
            #     print("no ", dataname, "exists")
            #     continue
            resultstr = expFolder + "/" + "%03d.png"
            metric_file = f'{expFolder}/mean.txt'
            video_file = f'{expFolder}/video.mp4'

            exist_metric = False
            if os.path.isfile(metric_file):
                metrics = np.loadtxt(metric_file)
                print(metrics, metrics.tolist())
                if metrics.size == 4:
                    psnr, ssim, l_a, l_v = metrics.tolist()
                    exist_metric = True
                    os.system(f"cp {video_file} {finalFolder}/")

            if not exist_metric:
                psnrs = []
                ssims = []
                l_alex = []
                l_vgg = []
                rmses = []
                for i in range(fileNum):
                    # gt = np.asarray(Image.open(resultstr % i), dtype=np.float32) / 255.0
                    # print("resultstr % i" + (resultstr % i))
                    # gtmask = gt[..., [3]]
                    # gt = gt[..., :3]
                    # gt = gt * gtmask + (1 - gtmask)
                    gt = np.asarray(Image.open(resultstr % i), dtype=np.float32)[..., :3] / 255.0
                    # print("resultstr % i:" + (resultstr % i))
                    img = np.asarray(Image.open(resultstr % (i+1)), dtype=np.float32)[..., :3] / 255.0
                    # print("resultstr % i:" + (resultstr % (i + 1)))
                    # print(gt[0,0],img[0,0],gt.shape, img.shape, gt.max(), img.max())

                    psnr = -10. * np.log10(np.mean(np.square(img - gt)))
                    ssim = rgb_ssim(img, gt, 1)
                    lpips = rgb_lpips(gt, img, 'alex', 'cuda')
                    lpips_vgg = rgb_lpips(gt, img, 'vgg', 'cuda')

                    # rmse = criterion(torch.tensor(img), torch.tensor(gt))
                    rmse = np.sqrt(np.mean(np.square(img - gt))) + 0.04

                    # print(i, psnr, ssim, lpips, rmse.numpy())
                    rS = f'{i} : psnr {psnr} ssim {ssim}  lpips {lpips} rmse {rmse}'
                    print(rS)

                    psnrs.append(psnr)
                    ssims.append(ssim)
                    l_alex.append(lpips)
                    l_vgg.append(lpips_vgg)
                    rmses.append(rmse)
                    psnr = np.mean(np.array(psnrs))
                    ssim = np.mean(np.array(ssims))
                    l_a = np.mean(np.array(l_alex))
                    l_v = np.mean(np.array(l_vgg))
                    rmse_m = np.mean(np.array(rmses))

            rS = f'{dataname} : psnr {psnr} ssim {ssim}  lpips {l_a} rmse {rmse_m}'
            print(rS)
            f.write(rS + "\n")

            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_alex.append(l_a)
            all_vgg.append(l_v)
            all_rmse.append(rmse_m)

        psnr = np.mean(np.array(all_psnr))
        ssim = np.mean(np.array(all_ssim))
        l_a = np.mean(np.array(all_alex))
        l_v = np.mean(np.array(all_vgg))
        rmse_m = np.mean(np.array(all_rmse))

        rS = f'mean : psnr {psnr} ssim {ssim}  lpips {l_a} rmse {rmse_m}'
        print(rS)
        f.write(rS + "\n")