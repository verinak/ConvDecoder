import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import time
import copy

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from bart import bart

import warnings
warnings.filterwarnings('ignore')

from include import *
from demo_helper.helpers import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print("num GPUs",torch.cuda.device_count())

def crop_kspace(kspace_data, target_width=640, target_height=320):
    """
    Crop k-space data from (4, 768, 392) to the target size (4, 640, 320).

    Parameters:
        kspace_data (numpy.ndarray): The k-space data to be cropped. Expected shape (4, 768, 392).
        target_width (int): The target width of the cropped k-space data (default is 640).
        target_height (int): The target height of the cropped k-space data (default is 320).

    Returns:
        numpy.ndarray: The cropped k-space data with shape (4, 640, 320).
    """
    kspace_shape = kspace_data.shape

    # Ensure the target size is smaller than or equal to the k-space dimensions
    assert target_width <= kspace_shape[1] and target_height <= kspace_shape[2], \
        "Target size must be smaller than or equal to k-space dimensions."

    # Calculate cropping ranges for the center of the k-space
    start_x = (kspace_shape[1] - target_width) // 2  # for width (768 -> 640)
    start_y = (kspace_shape[2] - target_height) // 2  # for height (392 -> 320)

    # Crop the k-space data (keeping the coil dimension intact)
    cropped_kspace = kspace_data[:, start_x:start_x + target_width, start_y:start_y + target_height]

    return cropped_kspace

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
torch.nn.Module.add = add_module

class conv_model(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, out_size, in_size):
        super(conv_model, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1]*(num_layers-1)

        ### compute up-sampling factor from one layer to another
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]

        ### hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers-1):

            self.net.add(nn.Upsample(size=hidden_size[i], mode='nearest'))
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True)
            self.net.add(conv)
            self.net.add(nn.ReLU())
            self.net.add(nn.BatchNorm2d( num_channels, affine=True))
        ### final layer
        self.net.add(nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True) )
        self.net.add(nn.ReLU())
        self.net.add(nn.BatchNorm2d( num_channels, affine=True))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))

    def forward(self, x, scale_out=1):
        return self.net(x)*scale_out
    
def get_scale_factor(net,num_channels,in_size,masked_kspace,ni=None):
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    if ni is None:
        shape = [1,num_channels, in_size[0], in_size[1]]
        ni = Variable(torch.zeros(shape)).type(dtype)
        ni.data.uniform_()
    # generate random image for the above net input
    out_chs = net( ni.type(dtype) ).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    out_img_tt = root_sum_of_squares( torch.tensor(out_imgs) , dim=0)

    ### get norm of zero-padded image
    orig_tt = ifft2(masked_kspace)        # Apply Inverse Fourier Transform to get the complex image
    orig_imgs_tt = complex_abs(orig_tt)   # Compute absolute value to get a real image
    orig_img_tt = root_sum_of_squares(orig_imgs_tt, dim=0)
    orig_img_np = orig_img_tt.cpu().numpy()

    ### compute scaling factor as norm(output)/norm(ground truth)
    s = np.linalg.norm(out_img_tt) / np.linalg.norm(orig_img_np)
    return s,ni

def create_network(num_channels, num_layers, in_size, kernel_size, output_depth, out_size, strides, init=True):

    net = conv_model(num_layers,num_channels, output_depth,out_size,in_size).type(dtype)

    if init:
      ### load the initialization
      state_url = '/content/initializations/brain_t2_init.pth'
      # state_url = '/initializations/brain_t2_init.pth'
      checkpoint = torch.load(state_url, map_location=torch.device("cpu"))  # Use "cuda" if loading to GPU
      # Load model weights safely (ignoring mismatched layers)
      checkpoint_dict = checkpoint['model_state_dict']
      model_dict = net.state_dict()

      # Keep only matching layer shapes
      filtered_checkpoint = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

      # Update model with matching layers
      model_dict.update(filtered_checkpoint)
      net.load_state_dict(model_dict)

      # Load net input
      ni = checkpoint['net_input'].type(dtype)

    else:
      # generate random input noise
      shape = [1,num_channels, in_size[0], in_size[1]]
      ni = Variable(torch.zeros(shape)).type(dtype)
      ni.data.uniform_();

    return net, ni

def get_sens_maps(masked_kspace, slice_ksp_torchtensor, mask2d):
  zpad = masked_kspace.data.cpu().numpy()
  zpad_complex = []
  for i in range(zpad.shape[0]):
      zpad_complex += [zpad[i,:,:,0]+1j*zpad[i,:,:,1]]
  zpad_complex = np.array(zpad_complex)
  masked_complex_kspace = zpad_complex * np.array(slice_ksp_torchtensor.shape[0]*[list(mask2d)]) # shape: (15, 640, 368)
  masked_kspace_np = np.array([np.moveaxis(masked_complex_kspace,0,2)])

  # Estimate sensitivity maps using BART ecalib
  sens_maps = bart(1, "ecalib -m1 -W -c0", masked_kspace_np)

  sens_maps_np = np.moveaxis(sens_maps[0],2,0)
  sens_maps_var = transform.to_tensor(sens_maps_np)

  return sens_maps_var

# with sensitivity maps

def forwardm(img_out,mask,sens_maps):
    ### apply coil sensitivity maps and forward model
    imgs = torch.zeros(sens_maps.shape).type(dtype)
    for j,s in enumerate(sens_maps):
        imgs[j,:,:,0] = img_out[0,0,:,:] * s[:,:,0] - img_out[0,1,:,:] * s[:,:,1]
        imgs[j,:,:,1] = img_out[0,0,:,:] * s[:,:,1] + img_out[0,1,:,:] * s[:,:,0]
    Fimg = transform.fft2(imgs[None,:])
    Fimg,_ = transform.apply_mask(Fimg, mask = mask)

    return Fimg

def fit(net, img_noisy_var, net_input, apply_f, mask, sens_maps, num_iter=5000, LR=0.01,verbose=True):
    net_input = net_input.type(dtype)
    p = [x for x in net.parameters()]

    mse_wrt_noisy = np.zeros(num_iter)

    optimizer = torch.optim.Adam(p, lr=LR)
    mse = torch.nn.MSELoss()

    best_net = copy.deepcopy(net)
    best_mse = float("inf")

    sens_maps = sens_maps.type(dtype) # here
    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            out = net(net_input.type(dtype))  # Network generates the final image

            # Apply forward model with sensitivity maps
            # el 7eta eli howa kateb 3aleha apply coil sensitivity maps di ana ha7otaha fl forwardm 34an ana moktane3a en heya heya
            loss = mse(apply_f(out, mask, sens_maps), img_noisy_var)
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()

            if i % 500 == 0:
                if verbose:
                  print(f'Iteration {i}    Train loss {loss.data}')

            return loss

        loss = optimizer.step(closure)

        if best_mse > 1.005 * loss.data:
            best_mse = loss.data
            best_net = copy.deepcopy(net)

    net = best_net
    return mse_wrt_noisy, net

def data_consistency(net, ni, mask1d, slice_ksp1, sens_maps):
    img_out = net(ni.type(dtype))  # Reconstructed image from the network
    # el mafroud yetla3li soura wa7da 3aks eli mn 8er sens maps eli bytala3 soura l kol coil w bye3melaha rss

    sens_maps = sens_maps.type(dtype) # here
    sh = sens_maps.shape

    # ana m4 ba3mel el line bta3 transform to tensor da 34an ana already badihalo var

    # tensors fadya 34an nfok fiha el sowar??
    # 4abah el 7eta eli kan bifok fiha el img l real w imaginary bs howa kan 3ando kaza img fa kan 3ando dimension/index zyada
    # a7ee
    imgs = torch.zeros(sh).type(dtype)
    for j,s in enumerate(sens_maps):
      # di complex multilplication 3adi
      imgs[j,:,:,0] = img_out[0,0,:,:] * s[:,:,0] - img_out[0,1,:,:] * s[:,:,1]
      imgs[j,:,:,1] = img_out[0,0,:,:] * s[:,:,1] + img_out[0,1,:,:] * s[:,:,0]

    # hamout wa3mel hena 7aga badal el ksp2measurement di bs ana m4 3arfa eh

    # Apply 2D Fourier Transform to get k-space representation
    Fimg = fft2(imgs[None, :])

    # slice_ksp1 has dim: (num_slices,x,y)
    # meas = slice_ksp_torchtensor1.unsqueeze(0) # dim: (1,num_slices,x,y,2)
    meas = ksp2measurement(slice_ksp1).type(dtype) # dim: (1,num_slices,x,y,2), azon
    meas = meas.detach().cpu() # um 34an error keda loh 3laka bl devices wl dtypes

    mask = torch.from_numpy(np.array(mask1d, dtype=np.uint8))
    ksp_dc = Fimg.clone()
    ksp_dc = ksp_dc.detach().cpu()
    ksp_dc[:,:,:,mask==1,:] = meas[:,:,:,mask==1,:] # after data consistency block

    img_dc = transform.ifft2(ksp_dc)[0]
    out = []
    for img in img_dc.detach().cpu():
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]

    par_out_chs = np.array(out)
    par_out_imgs = channels2imgs(par_out_chs)
    prec = root_sum_of_squares(torch.from_numpy(par_out_imgs)).numpy()
    if prec.shape[0] > 320:
        prec = crop_center(prec,320,320)

    # prec = crop_center2(root_sum_of_squares2(par_out_imgs),320,320)

    return prec

def brain_rec(slice_ksp, fully_sampled=True, verbose=True):

    if slice_ksp.shape[2] > 320:
      slice_ksp = crop_kspace(slice_ksp, target_width=640, target_height=320)
    
    numit = 10000
    LR = 0.01

    num_channels = 64
    num_layers = 5
    in_size = [8,4]
    kernel_size = 3
    output_depth = 2
    out_size = slice_ksp.shape[1:] # heya heya bs el tanya tensor fa 34an keda fi -1 azon

    strides = [1]*(num_layers-1)

    net, ni = create_network(
        num_channels = 64,
        num_layers = 5,
        in_size = [8,4],
        kernel_size = 3,
        output_depth = 2,
        out_size = slice_ksp.shape[1:],
        strides = [1]*(num_layers-1),
        init=True,
      )
    
    # convert to tensor
    data = slice_ksp.copy()
    data = np.stack((data.real, data.imag), axis=-1)
    slice_ksp_torchtensor = transform.to_tensor(slice_ksp)

    # reconstruct original image
    measurement = ksp2measurement(slice_ksp)
    lsimg = lsreconstruction(measurement).abs()
    lsrec = crop_center2(root_sum_of_squares2(var_to_np(lsimg)), 320, 320)

    # apply mask
    # da ta5ayoli l eli el mafroud ye7sal lw el soura m4 fully sampled 
    if fully_sampled:
        mask, mask1d, mask2d = get_mask(slice_ksp_torchtensor, slice_ksp)
        masked_kspace, _ = apply_mask(slice_ksp_torchtensor, mask = mask)
    else:
        masked_kspace = slice_ksp_torchtensor.clone() # copy

    # fix scaling for the decoder
    scaling_factor, ni = get_scale_factor(net,
                                  num_channels,
                                  in_size,
                                  masked_kspace,
                                  ni = ni,)
    masked_kspace *= scaling_factor
    unders_measurement = Variable(masked_kspace[None,:])

    # estimate sensitivity maps
    sens_maps_var = get_sens_maps(masked_kspace, slice_ksp_torchtensor, mask2d)

    # run reconstruction
    start = time.time()

    mse_wrt_noisy, net = fit(
        copy.deepcopy(net),
        unders_measurement.type(dtype),
        net_input=ni,
        apply_f=forwardm,
        mask=mask.type(dtype),
        sens_maps=sens_maps_var,
        num_iter=numit,
        LR=LR,
        verbose=verbose
    )

    # apply data consistency
    rec = data_consistency(net, ni, mask1d, scaling_factor * slice_ksp,sens_maps_var)

    if verbose:
      print('\nFinished after %.1f minutes.' % ((time.time() - start) / 60))

    return lsrec, rec