import toml
import pandas as pd
import torch
import numpy as np

# TODO: make everything singular, i.e. load_reflectances -> load_reflectance, ...

def load_config(filename):
    return toml.load(filename)

def load_reflectances(filenames):
    dfs = [pd.read_csv(filename, index_col=0).dropna(axis=1) for filename in  filenames]
    for df in dfs: df.columns = df.columns.astype(float)
    columns = [torch.tensor(df.columns) for df in dfs]        
    tensors = [torch.tensor(df[col].values) for df, col in zip(dfs, columns)]

    return dict(        
        columns = columns,
        data    = tensors,
    )

def load_photos(filenames):
    dfs = [pd.read_csv(filename, header=None) for filename in  filenames]
    tensors = [torch.tensor(df.values).float() for df in dfs]
    return tensors

def load_camera(filename):
    data = pd.read_csv(filename).values
    data = data if data.shape[0] > data.shape[1] else data.T

    return dict(
        columns = torch.tensor(data[:,0]).float(),
        data    = torch.tensor(data[:,1:]).float(),
    )

def load_camera_old(filename):
    df = pd.read_csv(filename).values
    print(df)
    df = df if df.shape[0] < df.shape[1] else df.T
    df.columns = df.columns.astype(float)
    
    return dict(
        data    = torch.tensor(df.T.values).float(),
        columns = torch.tensor(df.columns).float(),
    )

def load_light(filename):
    df         = pd.read_csv(filename, index_col=0).T
    df.columns = df.columns.astype(float)

    return dict(
        data    = torch.tensor(df.values).float().view(-1),
        columns = torch.tensor(df.columns).float(),
    )

def load_lights(filenames):
    dfs = [pd.read_csv(filename, index_col=0).T for filename in filenames]
    for df in dfs: df.columns = df.columns.astype(float)
    columns = [torch.tensor(df.columns) for df in dfs]
    tensors = [torch.tensor(df.values) for df in dfs]

    return dict(
        columns = columns,
        data    = tensors,
    )

def load_wavelengths(filename):
    df = pd.read_csv(filename, header=None).astype(float)
    return torch.tensor(df.values).view(-1).float()

def load_sfunctions(filename):
    df = pd.read_csv(filename).T

    return dict(
        columns = torch.tensor(df.iloc[0].values).float(),
        data = torch.tensor(df.iloc[1:].values).float(),
    )    

def load_cmatrices(filenames):
    dfs = [pd.read_csv(filename, header=None) for filename in  filenames]
    tensors = [
        torch.tensor(
            np.linalg.inv(df.values.T) # Transpose because of equation RLC = P
        ).float()
        for df in dfs]
    return tensors

def load_fmatrices(filenames):
    dfs     = [pd.read_csv(filename, header=None) for filename in filenames]
    tensors = [torch.tensor(df.values.T).float() for df in dfs]
    return tensors

def save_camera(filename, camera, wavelengths):
    data = torch.cat([wavelengths.view(-1,1), camera], axis=1).numpy()
    out_df = pd.DataFrame(data, columns=["wavelength", "red", "green", "blue"])
    out_df.to_csv(filename, index=False)
    
# Probably not needed ##########################################################
def load_illuminants(filename):
    df = pd.read_csv(filename)

    return dict(
        columns = torch.tensor(df.iloc[:,0].values).float(),
        data = torch.tensor(df.iloc[:,1:].values).float().T,
    )
