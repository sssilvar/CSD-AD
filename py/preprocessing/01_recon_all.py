import os
from multiprocessing import Pool


# Set dataset folder and extension files
dataset_folder = '/run/media/sssilvar/TOSHIBA EXT/Alzheimer/adni - imagenes nifti/SELECCIONADOS'
subjects_dir = ''
ext = '.nii'


def recon_all(files, dataset_folder, subjects_dir):
    """
    This function executes the command recon-all of FreeSurfer
    See doc at: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
    """
    for file_input in files:

        command = 'recon-all -i ' + os.path.join(dataset_folder, file_input) + ' -sd ' + subjects_dir\
                  + ' -s ' + file_input[:-4] + ' -all'
        print(command)


def get_file_list(path, ext):
    """
    :param path: Dataset path
    :param ext: Extension to be found
    :return: List of files in that directory
    """
    files = []
    for root, dirs, files in os.walk(path):
        if "*" + ext in dirs:
            files.append(dirs)

    return files


if __name__ == '__main__':
    files = get_file_list(dataset_folder, ext)

    # Paralleling process
    pool = Pool(6)
    pool.map(recon_all, (files, dataset_folder, ))

    recon_all(files, dataset_folder, 'subjects_dir')


