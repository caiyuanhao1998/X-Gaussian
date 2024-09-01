import torch
import cv2
import numpy as np
from tqdm import tqdm
from pdb import set_trace as stx

get_mse = lambda x, y: torch.mean((x - y) ** 2)

def get_psnr(x, y):
    if torch.max(x) == 0 or torch.max(y) == 0:
        return torch.zeros(1)
    else:
        x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        y_norm = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        mse = get_mse(x_norm, y_norm)
        psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(x.device))
    return psnr


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def alpha_to_density(opacity):
    density = -torch.log(1-opacity)
    return density

def psnr_3d(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def min_max_norm(tensor, normalize=True):
    # tensor range: [0, 1]
    # Conver to PIL Image and then np.array (output shape: (H, W))
    if normalize:
        img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return img

def cast_to_image(tensor, normalize=True):
    # tensor range: [0, 1]
    # Conver to PIL Image and then np.array (output shape: (H, W))
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor
    if normalize:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img[..., np.newaxis]

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename



def recover_covariance_matrix(uncertainty):
    # Number of matrices to reconstruct
    n = uncertainty.shape[0]

    # Initialize a tensor to store the reconstructed matrices
    L = torch.zeros((n, 3, 3), dtype=uncertainty.dtype, device=uncertainty.device)

    # Fill the diagonal and lower diagonal elements of L with elements from uncertainty
    L[:, 0, 0] = uncertainty[:, 0]
    L[:, 0, 1] = uncertainty[:, 1]
    L[:, 0, 2] = uncertainty[:, 2]
    L[:, 1, 0] = uncertainty[:, 1]  # Symmetric element
    L[:, 1, 1] = uncertainty[:, 3]
    L[:, 1, 2] = uncertainty[:, 4]
    L[:, 2, 0] = uncertainty[:, 2]  # Symmetric element
    L[:, 2, 1] = uncertainty[:, 4]  # Symmetric element
    L[:, 2, 2] = uncertainty[:, 5]

    return L
