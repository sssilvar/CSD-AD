import os


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


def mkdir(dir):
    """
    Creates a folder
    :param dir: complete path to the folder to be created.
    :return: None
    """
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except IOError:
            print('[  WARNING  ] Folder could not be created')
