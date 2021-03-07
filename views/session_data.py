'''
    Class session data file  
    
    MEJORAR TODO ESTO!!!!!!!!!!
'''
import numpy as np

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




#MEJOR ASII
def session_data_dict():

        session_data = {}

        #Dataset
        session_data['n_samples'] = 0
        session_data['n_features'] = 0
        session_data['discrete_data'] = False
        session_data['columns_names'] = []

        

        #SOM Params
        session_data['som_tam_eje_x'] = 0 
        session_data['som_tam_eje_y'] = 0 

        return session_data          