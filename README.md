# Spectral sensitivity estimation without a camera

This repository contains the code for paper [Spectral sensitivity estimation without a camera](https://arxiv.org/abs/2304.11549).

A pretrained autoencoder can be found in `output/autoencoder.pt`. To retrain it, you can run the following command in a Linux terminal:
```shell
bash run-autoencoder.sh
```

To estimate spectral sensitivities for Canon EOS 10D, you can run:
```shell
bash run-camest.sh 
```

More than 1000 spectral sensitivity predictions can be found in `data/predictions`.

## Command-line interface (Linux)

### Training the autoencoder
The content of `run-autoencoder.sh` is

```shell
python autoencoder.py \
       --config ./configs/autoencoder.toml \
       --train_cameras ./data/ground-truths/*.csv \
       --test_cameras ./data/ground-truths/test-data/*.csv \
       --wavelengths ./data/wavelengths.csv \
       --model ./output/autoencoder.pt \
```

The meaning of the arguments is listed below:
```
--config: Configuration file
--train_cameras: Spectral sensitivities used for training
--test_cameras: Spectral sensitivities used for testing
--wavelengths: Discretization of wavelengths (used for potential interpolation)
--model: Where to save the autoencoder after training
```

### Estimating spectral sensitivities

The content of `run-camest.sh` is

```
python camest.py est \
       --config ./configs/camest.toml \
       --wavelengths ./data/wavelengths.csv \
       --autoencoder ./output/autoencoder.pt \
       --observer ./data/cie-standard-observer.csv \
       --cmatrices ./data/color-matrices/Canon-EOS-10D-cm*.csv \
       --cilluminants ./data/illuminants/illuminant-C*.csv \
       --predicted ./output/predicted-Canon-EOS-10D.csv
```

The meaning of the arguments is listed below:

```
--config: Configuration file \
--wavelengths: Discretization of wavelengths (used for potential interpolation) \
--autoencoder: Path of the pretrained autoencoder \
--observer: Path to CIE-standard observer \
--cmatrices: Paths for the two color matrices \
--cilluminants Paths for the two corresponding illuminants \
--predicted: Where to save the predicted spectral sensitivities
```

## Contact info

For questions, suggestions, contributions etc., please contact Grigory Solomatov: <gsolomat@campus.haifa.ac.il>.
