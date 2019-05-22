
# Spherical curvelet representation
Here it is the pipeline taken into account:

 1. `feature_extraction.py`: Goes over all the data set (already mapped to spheres with `py/04_mapping_sphere/radial_multiscale_analysis.py`) and performs a Curvelet decomposition. The results are saved in `output/curv_feats_[intensity,gradient]_nscales_#_nangles_#/[subject_id].npz`.
 2. `create_dataframe.py`: Goes over the output of the step 1. As output, a DataFrame in HDF5 format with the same name of the folder.
