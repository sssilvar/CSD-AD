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