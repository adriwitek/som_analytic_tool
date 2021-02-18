'''
    Class session data file  
    
    MEJORAR TODO ESTO!!!!!!!!!!
'''


class Sesion():
    
    data = None     # numpy array
    n_samples = 0
    n_features = 0 
    #tipo de modelo
    modelo = None

    def __init__(self,data):
        self.data = data
        self.n_samples, self.n_features=data.shape
        self.modelo = None

    def set_modelo(self,modelo):
        self.modelo = modelo
    def get_modelo(self):
        return self.modelo