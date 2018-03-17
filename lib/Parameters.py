import json


class Parameters:

    def __init__(self, params_file):
        self.params_file = params_file
        self.jf = self.extract_params()

        self.dataset_folder = self.jf['dataset_folder']
        self.data_file = self.jf['data_file']

    def extract_params(self):
        with open(self.params_file) as json_file:
            print('[  OK  ] Loading %s' % self.params_file)
            jf = json.load(json_file)
        return jf
