# Crop image for training
python scripts/data_preparation/extract_bayer_subimages_with_metadata.py \
    --data-path datasets/ICCV23-LED/Sony/long \
    --save-path datasets/ICCV23-LED/Sony_train_long_patches \
    --suffix ARW \
    --n-thread 10

# Convert the ELD data into SID data structure
python scripts/data_preparation/eld_to_sid_structure.py \
    --data-path datasets/ICCV23-LED/ELD_sym \
    --save-path datasets/ICCV23-LED/ELD

# convert SID SonyA7S2
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/Sony --save-path datasets/ICCV23-LED/Sony_npy --suffix ARW --n-thread 8
# convert ELD SonyA7S2
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/SonyA7S2 --save-path datasets/ICCV23-LED/ELD_npy/SonyA7S2 --suffix ARW --n-thread 8
# convert ELD NikonD850
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/NikonD850 --save-path datasets/ICCV23-LED/ELD_npy/NikonD850 --suffix nef --n-thread 8
# convert ELD CanonEOS70D
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/CanonEOS70D --save-path datasets/ICCV23-LED/ELD_npy/CanonEOS70D --suffix CR2 --n-thread 8
# convert ELD CanonEOS700D
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/CanonEOS700D --save-path datasets/ICCV23-LED/ELD_npy/CanonEOS700D --suffix CR2 --n-thread 8