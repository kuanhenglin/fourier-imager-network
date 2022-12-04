#!/bin/sh

cp main.py main_run.py  #further (potentially breaking) modifications to main.py will not halt run

#FOURIER RESNET
python3 main_run.py -t fourier_resnet -l 0.1 -e 0
python3 main_run.py -t fourier_parallel -l 0.1 -e 0
python3 main_run.py -t fourier_skip -l 0.1 -e 0

#VANILLA RESNET
python3 main_run.py -t fourier_resnet -l 0.1 -f 0 0 0 -s 0 -e 0  #ResNet-20
python3 main_run.py -t fourier_resnet -l 0.1 -r 5 -f 0 0 0 0 0 -s 0 -e 0  #ResNet-32
