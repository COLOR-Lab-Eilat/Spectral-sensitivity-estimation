# Third party imports
import camest
import argparse
from pathlib import Path

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Internal imports
import loader
import models

from tensortools import normalize_max, interp

# Loading ######################################################################
def load_data_std(cli_args):
    data = load_data(cli_args)
    wavelengths = data["wavelengths"]    

    # Interpolate
    interp_cameras = lambda cameras : [
        interp(wavelengths, camera["columns"], camera["data"].T).T
        for camera in cameras
    ]
    
    train_cameras = interp_cameras(data["train_cameras"])
    test_cameras = interp_cameras(data["test_cameras"])   

    # Normalize
    normalize_cameras = lambda cameras : torch.stack([normalize_max(camera) for camera in cameras], dim=0)

    train_cameras = normalize_cameras(train_cameras)
    test_cameras = normalize_cameras(test_cameras)

    return dict(
        config        = data["config"],
        model         = data["model"],
        wavelengths   = wavelengths,
        train_cameras = train_cameras,
        test_cameras  = test_cameras,
    )

def load_data(cli_args):
    try:
        assert 1 == 2, ""
        model = torch.load(cli_args["model"])        
        print(f"Loaded {cli_args['model']}")
    except:
        model = None
        print(f"No existing model found, creating a new one")

    return dict(
        config        = loader.load_config(cli_args["config"]),
        wavelengths   = loader.load_wavelengths(cli_args["wavelengths"]),
        train_cameras = [loader.load_camera(filename) for filename in cli_args["train_cameras"]],
        test_cameras  = [loader.load_camera(filename) for filename in cli_args["test_cameras"]],
        model         = model,
    )

# Optimization #################################################################
def create_optims(data):
    config = data["config"]
    model  = models.Autoencoder(
        inputs           = torch.numel(data["train_cameras"][0]),
        bottleneck       = config["bottleneck"],        
        encoder_sizes    = config["encoder_sizes"],
        decoder_sizes    = config["decoder_sizes"],        
        encoder_dropouts = config["encoder_dropouts"],
        decoder_dropouts = config["decoder_dropouts"],
    ) if data["model"] is None else data["model"]
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config["learning_rate"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                   factor=config["scheduler_decay"],
                                   patience=config["patience"])

    train_dataloader = torch.utils.data.DataLoader(
        data["train_cameras"],
        batch_size = data["train_cameras"].shape[0],
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        data["test_cameras"],
        batch_size = data["test_cameras"].shape[0],
        shuffle=True,
    )

    # Good
    def loss_fn(cams_in, cams_out):
        serrors = (cams_in - cams_out)/cams_in.norm(dim=1, keepdim=True)
        serrors = serrors.view(serrors.shape[0], -1)
        norms = serrors.norm(dim=1)
        return norms.mean()
    
    return dict(
        config           = config,
        model            = model,
        optimizer        = optimizer,
        scheduler        = scheduler,
        train_dataloader = train_dataloader,
        test_dataloader  = test_dataloader,
        loss_fn          = loss_fn,
    )

def get_validation_loss(optims):
    optims["model"].eval()
    
    loss = 0.0
    for cams_in in optims["test_dataloader"]:
        cams_out = optims["model"](cams_in)        
        loss += optims["loss_fn"](cams_in, cams_out)

    optims["model"].train()
    return loss

def run_epoch(optims):
    train_loss = 0.0
    for cams_in in optims["train_dataloader"]:

        # Augumentation ########################################################
        cams_in *= 1.0 - optims["config"]["scaling"]*torch.rand(cams_in.shape[0], 1, cams_in.shape[2])
        cams_in /= cams_in.view(cams_in.shape[0],-1).max(dim=1)[0].view(-1, 1, 1)

        roll = optims["config"]["roll"]
        for cam_idx in range(cams_in.shape[0]):
            for channel_idx in range(cams_in.shape[2]):
                cams_in[cam_idx,:,channel_idx] = torch.roll(
                    cams_in[cam_idx,:,channel_idx],
                    torch.randint(-roll,roll+1,(1,)).item(),
                )
               
        ########################################################################

        cams_out = optims["model"](cams_in)
        loss     = optims["loss_fn"](cams_in, cams_out)
        
        train_loss += loss.item()
        
        optims["optimizer"].zero_grad()
        loss.backward()
        optims["optimizer"].step()

    test_loss = get_validation_loss(optims).item()
    optims["scheduler"].step(train_loss)

    return train_loss, test_loss

# Main #########################################################################
def get_cli_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--train_cameras", nargs="+", type=str, help="Camera files for training")
    parser.add_argument("--test_cameras", nargs="+", type=str, help="Camera files for testin")
    parser.add_argument("--wavelengths", type=str, help="Wavelength file")
    parser.add_argument("--model", type=str, help="Where to save model")
    parser.add_argument("--plot", type=str, help="Where to save plot")
    
    return vars(parser.parse_args())

def single_main(cli_args):
    data     = load_data_std(cli_args)    
    config   = data["config"]
    optims   = create_optims(data)
    
    num_params = sum(p.numel() for p in optims["model"].parameters())
    print(optims["model"])
    #print(f"Parameters: {num_params}")

    loss_history = dict(
        train = [],
        test  = [],
    )

    counter = 5
    for epoch in range(int(1e+6)):                
        train_loss, test_loss = run_epoch(optims)

        if epoch % 10 == 0:
            print("Grad step: {}, loss: {:.4f}".format(epoch, train_loss), end="\r")
        
        loss_history["train"].append(train_loss)
        loss_history["test"].append(test_loss)

        if optims["optimizer"].param_groups[0]['lr'] < optims["config"]["stop_thresh"]: break        

    print("Finished")
    torch.save(optims["model"], cli_args["model"])

    return dict(
        data   = data,
        optims = optims,
        config = config,
    )

def main():
    cli_args = get_cli_input()
    single_main(cli_args)       

if __name__ == "__main__": main()
