import sys
from mne.channels import read_dig_montage
from mne.io import read_raw_egi


# Command Line Arguments
# 1. The path to the EGI simple binary (.raw) file
# 2. The path to the corresponding coordinates (.xml) file
# 3. The output path to where you'd like to write the fif file
# Note that the output fif file should end with _raw.fif to keep with MNE
# convetion

# Example of usage:
# python convert_egi_raw.py ./egi_raw/test_data.raw ./coordinates/
# test_data_coordinates.xml ./fif_raw/test_raw.fif
arguments = sys.argv
raw_input_path = arguments[1]
coordinates_path = arguments[2]
raw_output_path = arguments[3]

raw = read_raw_egi(raw_input_path, misc=['EEG 257'])
dig_montage = read_dig_montage(egi=coordinates_path)
raw.set_montage(dig_montage)

raw.save(raw_output_path)
