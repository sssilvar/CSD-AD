import os


class LoadDataset(object):

    # Atributes
    list_of_files = []

    def __init__(self, f_path, csv_file, ext='mgz'):
        self.f_path = f_path
        self.csv_file = csv_file
        self.ext = ext

    def find_files(self):
        files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(data_path)
                 for name in files
                 if name.endswith(self.ext)]
        return files
