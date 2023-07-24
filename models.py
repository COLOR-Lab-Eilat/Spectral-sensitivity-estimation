import torch
from torch import nn

from tensortools import normalize, normalize_max

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        inputs        = kwargs["inputs"]
        bottleneck    = kwargs["bottleneck"]
        encoder_sizes = kwargs["encoder_sizes"]
        decoder_sizes = kwargs["decoder_sizes"]

        encoder_dropouts = kwargs["encoder_dropouts"] + [0.0] # No dropout before latent
        decoder_dropouts = kwargs["decoder_dropouts"] + [0.0] # No dropout before output

        encoder_sizes = torch.cat([
            torch.tensor(inputs).view(1),
            inputs*torch.tensor(encoder_sizes),
            torch.tensor(bottleneck).view(1),
        ]).ceil().int()
        decoder_sizes = torch.cat([
            torch.tensor(bottleneck).view(1),
            inputs*torch.tensor(decoder_sizes),
            torch.tensor(inputs).view(1),
        ]).ceil().int()

        triplets = lambda layer_sizes, dropouts : [
            (nn.Linear(inputs, outputs), nn.Dropout(p=dropout), nn.ReLU())
            for inputs, outputs, dropout in zip(layer_sizes[0:-1], layer_sizes[1:], dropouts)
        ]

        seq = lambda layer_sizes, dropouts : [el for triplet in triplets(layer_sizes, dropouts) for el in triplet]

        self.flatten = nn.Flatten()
        # Below: [:-2] to remove last dropout and activation
        self.encoder = nn.Sequential(*seq(encoder_sizes, encoder_dropouts)[:-2])
        self.decoder = nn.Sequential(*seq(decoder_sizes, decoder_dropouts)[:-2])        

    def forward(self, x):
        shape = x.shape
                
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x.view(*shape)

class CameraEstimator(nn.Module):
    @staticmethod
    def camera_init(num_wavelenths):
        peak_offset = 0.1
        sigma = 0.3
        x = torch.linspace(0, 1, num_wavelenths)        
        
        gaussian = lambda peak : torch.exp(-torch.pow((x - peak)/sigma, 2))

        camera = torch.stack([
            gaussian(0.4 + peak_offset*i) for i in [1,0,-1]
        ], dim = 1)

        return normalize(camera)

    def __init__(self, data):
        super().__init__()

        # Data #################################################################
        self.config       = data["config"]
        self.wavelengths  = data["wavelengths"]
        self.autoencoder  = data["autoencoder"]
        self.observer     = data["observer"]
        self.cmatrices    = data["cmatrices"]
        self.cilluminants = data["cilluminants"]
        
        self.delta        = self.wavelengths[1]-self.wavelengths[0]

        # Parameters ###########################################################
        self.camera = nn.Parameter(CameraEstimator.camera_init(len(self.wavelengths)))

    def get_normalized_camera(self):
        camera = normalize(torch.abs(self.camera))
        return camera

    def get_autoencoded_camera(self):
        camera = self.camera#/self.camera.max()
        acamera = self.autoencoder(torch.unsqueeze(camera, dim=0))
        acamera = torch.squeeze(acamera)
        return normalize(torch.abs(acamera))

    def forward(self):
        camera = self.get_normalized_camera()
        acamera = self.get_autoencoded_camera()
        
        # Loss terms ###########################################################                
        autoencoder_error = torch.acos(
            torch.clamp(                
                torch.dot(camera.flatten(), acamera.flatten())/(camera.norm()*acamera.norm() + 1e-7),
                min = -1.0 + 1e-7,
                max = 1.0 - 1e-7,
            )            
        )

        # C-matrices
        predicted_cmatrices = [torch.linalg.lstsq(L.T*camera, L.T*self.observer).solution
                               for L in self.cilluminants]
        cerror = sum(
                torch.acos(
                    torch.clamp(
                        (torch.dot(pred_cmat.flatten(), cmat.flatten()))/(pred_cmat.norm()*cmat.norm() + 1e-7),
                        min = -1.0 + 1e-7,
                        max = 1.0 - 1e-7,
                    ),                
                
            )
            for cmat, pred_cmat in zip(self.cmatrices, predicted_cmatrices)
        )

        norm_error = (self.camera.norm() - 1.0)**2

        loss = sum([
            self.config["autoencoder"]*autoencoder_error,
            self.config["cmatrices"]*cerror,
        ])        
               
        return loss
