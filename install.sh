# setting conda hook
eval "$(conda shell.bash hook)"
# create env
conda create -y -n LED-ICCV23 python=3.8
conda activate LED-ICCV23
# install packages
pip install -r requirements.txt
python setup.py develop

### install rawpy and LibRaw for benchmark
mkdir -p downloads/
# download LibRaw and rawpy
python scripts/download_gdrive.py --id 1B4gJYe3h4UxWTMZzNL2xVodSDUlNbeAR --save-path downloads/LibRaw-0.19.1.zip
python scripts/download_gdrive.py --id 1EuJsbZ_a_YJHHcGAVA9TXXPnGU90QoP4 --save-path downloads/rawpy.zip

unzip downloads/rawpy.zip -d downloads/
unzip downloads/LibRaw-0.19.1.zip -d downloads/

# setting LibRAW
cd downloads/LibRaw-0.19.1
./configure
make

# setting rawpy
cd ../rawpy
pip install -e .
