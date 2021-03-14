'''
    Class session data file  
    
    MEJORAR TODO ESTO!!!!!!!!!!
'''
import numpy as np
import json
import pickle

from  config.config import *




class Sesion():
    
    

    data = None     # numpy array
    n_samples = 0
    n_features = 0 
    #tipo de modelo
    modelo = None

    def __init__(self,data):
        data = np.copy(data)
        self.n_samples, self.n_features=data.shape
        self.modelo = None

    def get_data(self):
        return self.data
    def set_modelo(self,modelo):
        self.modelo = modelo
    def get_modelo(self):
        return self.modelo

    




    @staticmethod
    def session_data_dict():

            session_data = {}

            #Dataset
            session_data['n_samples'] = 0
            session_data['n_features'] = 0
            session_data['discrete_data'] = True
            session_data['columns_names'] = []



            #SOM Params
            session_data['som_tam_eje_x'] = 0 
            session_data['som_tam_eje_y'] = 0 

            return session_data          

    @staticmethod
    def get_nombres_atributos():

        with open(SESSION_DATA_FILE_DIR) as json_file:
            session_data = json.load(json_file)

        nombres = session_data['columns_names']
        atribs= nombres[0:len(nombres)-1]
        options = []  # must be a list of dicts per option

        for n in atribs:
            options.append({'label' : n, 'value': n})

        return options

    @staticmethod
    def save_model(model,filename):
        with open('filename', 'wb') as outfile:
            pickle.dump(model, outfile)


    @staticmethod
    def load_model(filename):
        with open('filename', 'rb') as infile:
            model = pickle.load(infile)
        return model