# Third party imports 
import argparse
import torch
from torch import nn
import numpy as np

from pathlib import Path

# Internal imports
import loader
from models import Autoencoder, CameraEstimator
from tensortools import interp, normalize, normalize_max
import numpy as np

# Loading ######################################################################
def load_data_std(file_dict):
    data = load_data(file_dict)
    wavelengths = data["wavelengths"]

    # Interpolate ##########################################################
    observer     = interp(wavelengths, data["observer"]["columns"], data["observer"]["data"].T).T
    cilluminants = [
        interp(wavelengths, columns, cilluminant)
        for columns, cilluminant in zip(data["cilluminants"]["columns"], data["cilluminants"]["data"])
    ]
        
    # Normalize data #######################################################
    observer     = normalize(observer)
    cilluminants = [normalize(illuminant) for illuminant in cilluminants]
    
    return dict(
        config       = data["config"],
        wavelengths  = wavelengths,
        observer     = observer,
        cilluminants = cilluminants,
        cmatrices    = data["cmatrices"],
        autoencoder  = data["autoencoder"],
    )

def load_data(file_dict):
    autoencoder = torch.load(file_dict["autoencoder"])    
    autoencoder.eval()
    
    return dict(
        config       = loader.load_config(file_dict["config"]),
        wavelengths  = loader.load_wavelengths(file_dict["wavelengths"]),
        observer     = loader.load_camera(file_dict["observer"]),
        cilluminants = loader.load_lights(file_dict["cilluminants"]),
        cmatrices    = loader.load_cmatrices(file_dict["cmatrices"]),
        autoencoder  = autoencoder,
    )

# Errors #######################################################################
def relative_full_scale_error(cam_true, cam_approx):
    rmse = torch.sqrt(
        torch.mean((cam_true - cam_approx)**2, dim=0)
    )
    return rmse/cam_true.max(dim=0)[0]*100

# Optimization #################################################################
def create_optims(data):
    config = data["config"]
    model  = CameraEstimator(data)
    
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=config["learning_rate"],
                    momentum=config["momentum"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                   factor=config["scheduler_decay"],
                                   patience=config["patience"])
    return dict(
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
        data      = data, # TODO: This was added late, probably makes many function args redundant
    )

def grad_step(optims):
    histories = dict(
        loss = [],
    )
    
    loss = optims["model"]()
    optims["optimizer"].zero_grad()
    loss.backward()
    
    optims["optimizer"].step()
    optims["scheduler"].step(loss.item())
        
    histories["loss"].append(loss.item())

    # Errors
    cam_approx = optims["model"].get_normalized_camera()
        
    return histories

# Main #########################################################################
def get_cli_input_root():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    get_cli_input_single(subparsers)

    args = parser.parse_args()
    args.func = args.func if hasattr(args, "func") else lambda args : None
    args.func(args)

def get_cli_input_single(subparsers):
    parser = subparsers.add_parser("est", add_help=False)
    
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--wavelengths", type=str, help="Wavelength file")
    parser.add_argument("--autoencoder", type=str, help="Autoencoder model")
    parser.add_argument("--cilluminants", nargs="+", type=str, help="Calibration illuminants")
    parser.add_argument("--observer", type=str, help="CIE XYZ standard observer color matching functions")
    parser.add_argument("--cmatrices", nargs="+", type=str, help="Calibration matrices 1", default=[])
    parser.add_argument("--predicted", type=str, help="Predicted folder")

    parser.set_defaults(func = lambda args : main_single(vars(args)))

def main_single(cli_args):
    data     = load_data_std(cli_args)
    optims   = create_optims(data)
    
    config = data["config"]

    histories = dict(
        loss = [],
        re  = [], # ground truth error
    )
    
    for i in range(int(1e+10)):
        new_histories = grad_step(optims)
        histories["loss"] += new_histories["loss"]

        if i % 100 == 0:
            print("Grad steps: {}, loss: {:.4f}".format(i, histories['loss'][-1]), end="\r")

        if optims["optimizer"].param_groups[0]["lr"] < config["stop_thresh"]:
            break

    camera = optims["model"].get_normalized_camera().detach()
    
    loader.save_camera(
        cli_args["predicted"],
        camera,
        data["wavelengths"],
    )
    print("Created {}".format(cli_args["predicted"]))

if __name__ == "__main__": get_cli_input_root()
