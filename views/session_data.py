'''
    Class session data file  
    
    MEJORAR TODO ESTO!!!!!!!!!!
'''
import numpy as np
import json
import pickle

from  config.config import *




class Sesion():
    
    
    #Dataset
    discrete_data = True
    file_data = None
    data = None     # numpy array
    n_samples = 0
    n_features = 0 

    #tipo de modelo
    modelo = None
    som_params= None
    gsom_params=None
    ghsom_params=None

    def __init__(self):
        return

    def set_data(self,data):
        self.data = np.copy(data)
        self.n_samples, self.n_features=data.shape

    def get_data(self):
        return self.data


    
    def set_filedata(self,filedata):
        self.file_data = filedata

    def get_filedata(self):
        return self.file_data 

    def set_modelo(self,modelo):
        self.modelo = modelo

    def get_modelo(self):
        return self.modelo

    def set_discrete_data(self,bool):
        self.discrete_data = bool

    def get_discrete_data(self):
        return self.discrete_data
    



    def set_som_model_info_dict(self,tam_eje_vertical,tam_eje_horizontal,learning_rate,neigh_fun,distance_fun,sigma,iteraciones, inicialitacion_pesos):

        #Only one model could be used at once
        self.som_params= {}
        self.gsom_params=None
        self.ghsom_params=None

        self.som_params['tam_eje_vertical'] = tam_eje_vertical
        self.som_params['tam_eje_horizontal'] = tam_eje_horizontal 
        self.som_params['learning_rate'] = learning_rate
        self.som_params['neigh_fun'] = neigh_fun
        self.som_params['distance_fun'] = distance_fun
        self.som_params['sigma'] = sigma
        self.som_params['iteraciones'] = iteraciones
        self.som_params['inicialitacion_pesos'] = inicialitacion_pesos
            
    def set_som_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= dict.copy()
        self.gsom_params = None
        self.ghsom_params=None

    def get_som_model_info_dict(self):
        return  self.som_params





    def set_gsom_model_info_dict(self,tam_eje_vertical,tam_eje_horizontal,tau_2,learning_rate,decadency,sigma,epocas_gsom,max_iter_gsom):

        #Only one model could be used at once
        self.som_params= None
        self.gsom_params={}
        self.ghsom_params=None

        self.gsom_params['tam_eje_vertical'] = tam_eje_vertical
        self.gsom_params['tam_eje_horizontal'] =tam_eje_horizontal
        self.gsom_params['tau_2'] = tau_2 
        self.gsom_params['learning_rate'] = learning_rate
        self.gsom_params['decadency'] = decadency
        self.gsom_params['sigma'] = sigma
        self.gsom_params['epocas_gsom'] = epocas_gsom
        self.gsom_params['max_iter_gsom'] = max_iter_gsom


    def set_gsom_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= None
        self.gsom_params = dict.copy()
        self.ghsom_params=None
            

    def get_gsom_model_info_dict(self):
        return  self.gsom_params







    def set_ghsom_model_info_dict(self,tau_1,tau_2,learning_rate,decadency,sigma,epocas_ghsom,max_iter_ghsom):

        #Only one model could be used at once
        self.som_params= None
        self.gsom_params=None
        self.ghsom_params={}    
        self.ghsom_params['tau_1'] = tau_1 
        self.ghsom_params['tau_2'] = tau_2 
        self.ghsom_params['learning_rate'] = learning_rate
        self.ghsom_params['decadency'] = decadency
        self.ghsom_params['sigma'] = sigma
        self.ghsom_params['epocas_gsom'] = epocas_ghsom
        self.ghsom_params['max_iter_gsom'] = max_iter_ghsom


    def set_ghsom_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= None
        self.gsom_params = None
        self.ghsom_params=dict.copy()
            

    def get_ghsom_model_info_dict(self):
        return  self.ghsom_params


    @staticmethod
    def session_data_dict():

            session_data = {}

            #Dataset
            session_data['n_samples'] = 0
            session_data['n_features'] = 0
            session_data['discrete_data'] = True
            session_data['columns_names'] = []



            #SOM Params
            session_data['som_tam_eje_vertical'] = 0 
            session_data['som_tam_eje_horizontal'] = 0 

            return session_data        


    #TODO CAMBIAR ESTO:BORRARLO
    @staticmethod
    def get_model_info_dict(model_type):

            model_info = {}

            #TODO   COMPLETAR ESTO

            if model_type == 'som':
                model_info['model_type'] = 'som'
                model_info['mapa_tam_eje_vertical'] = 0 
                model_info['mapa_tam_eje_horizontal'] = 0 
            elif model_type == 'gsom':
                model_info['model_type'] = 'gsom'
                model_info['mapa_tam_eje_vertical'] = 0 
                model_info['mapa_tam_eje_horizontal'] = 0 
            elif model_type == 'ghsom':
                model_info['model_type'] = 'ghsom'
            else: 
                return {}

            return model_info     



     
        

    @staticmethod
    def get_nombres_atributos():

        with open(SESSION_DATA_FILE_DIR) as json_file:
            datos_entrenamiento = json.load(json_file)

        nombres = datos_entrenamiento['columns_names']
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


session_data = Sesion()