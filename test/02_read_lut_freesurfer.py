import os
import pandas as pd

# Set root folder
root = os.path.join(os.getcwd(), '..')
lut_filename = os.path.join(root, 'param', 'FreeSurferColorLUT.txt')
file_out = os.path.join(os.path.dirname(lut_filename), 'FreeSurferColorLUT.csv')

# Load file (multiples spaces as separator)
regions_df = pd.read_csv(lut_filename, comment='#', index_col=False, sep='\s*', header=None, engine='python')
regions_df.columns = ['region_id', 'label_name', 'color_r', 'color_g', 'color_b', 'a']
print(regions_df.head(2))

regions_df.to_csv(file_out)
