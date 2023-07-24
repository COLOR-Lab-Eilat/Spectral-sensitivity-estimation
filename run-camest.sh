python camest.py est \
       --config ./configs/camest.toml \
       --wavelengths ./data/wavelengths.csv \
       --autoencoder ./output/autoencoder.pt \
       --observer ./data/cie-standard-observer.csv \
       --cmatrices ./data/color-matrices/Canon-EOS-10D-cm*.csv \
       --cilluminants ./data/illuminants/illuminant-C*.csv \
       --predicted ./output/predicted-Canon-EOS-10D.csv
