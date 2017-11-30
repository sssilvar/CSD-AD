import lib.Dataset as ds

data_path = '/home/sssilvar/Documents/datasets/RAW_preprocessed/ADNI_images'
ext = 'mgz'

data = ds.Dataset(f_path=data_path, csv_file='', ext=ext)
file_list = data.find_files()

print file_list


# Load Data
# Leave one out
# Create a class model
# Calculate
