import json
from tkinter import N

class Params:
    def __init__(self, path=None):
        self.__path = path
        self.__params = None

        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                self.__params = json.load(f)

    def from_dict(self, pars):
        self.__params = pars
        return self
    
    def get(self, name):
        if name not in self.__params:
            raise ValueError(f"Parameter with name {name} is not defined in {self.__path}")

        return self.__params[name]

    def params(self):
        return self.__params

    def override(self, name, value):
        if name not in self.__params:
            raise ValueError(f"Parameter with name {name} is not defined in {self.__path}")
        
        self.__params[name] = value
    

    def save(self, path):

        with open(path, "w") as f:
            json.dump(self.__params, f, indent=4)
