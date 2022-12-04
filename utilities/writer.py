from datetime import datetime
import pickle, json


class Writer():

    def __init__(self, categories, hyperparameters, prefix="run"):
        self.data = {}
        for category in categories:
            self.data[category] = []
        self.hyperparameters = hyperparameters
        self.name = prefix + "_" + datetime.now().strftime("%y%m%d_%H%M%S")

    def add(self, info):
        for category in info:
            self.data[category].append(info[category])

    def save(self, folder="logs", save_pickle=True, save_json=True):
        data_save = dict(data=self.data, hyperparameters=self.hyperparameters)
        if save_pickle:
            with open(folder + "/" + self.name + ".pickle", 'wb') as handle:
                pickle.dump(data_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if save_json:
            with open(folder + "/" + self.name + ".json", "w") as handle:
                json.dump(data_save, handle)