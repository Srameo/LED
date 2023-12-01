# Noise calibration

## Prerequisites 

### 1. Rawpy

The calibration code the raspy package, which can be installed based on this instructions(https://pypi.org/project/rawpy/)

Note: If rawpy cannot be installed on Mac(m1/m2), you can refer to this issue(https://github.com/letmaik/rawpy/issues/171)

### 2. data

It is recommended that you organize the data to be calibrated into  separate folders for each camera. Within each camera folder, you should  further divide the data into several subfolders based on different ISO  values.

## calibration process

For the calibration process, you can execute all steps at once by  directly following the code given in the main function, or you can  perform each step separately according to your needs. These steps  include selecting the positions of color blocks, calibrating the color  blocks to obtain K, calibrating dark images to obtain other parameters, and fitting log(K) and log(variance).