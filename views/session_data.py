'''
    Class session data file  
    
    MEJORAR TODO ESTO!!!!!!!!!!
'''
import numpy as np
import pickle

from  config.config import *




class Sesion():
    
    
  

    def __init__(self):
        #Dataset
        self.discrete_data = True
        self.file_data = None
        self.dataset = None     # numpy array
        self.n_samples = 0
        self.n_features = 0 
        self.columns_names= []

        #tipo de modelo
        self.modelo = None
        self.som_params= None
        self.gsom_params=None
        self.ghsom_params=None
        self.ghsom_structure_graph = {}
        self.ghsom_nodes_by_coord_dict = {}
        return


    #Call when closing app
    def clean_session_data(self):
        #Dataset
        self.discrete_data = True
        self.file_data = None
        self.data = None     # numpy array
        self.n_samples = 0
        self.n_features = 0 

        #tipo de modelo
        self.modelo = None
        self.som_params= None
        self.gsom_params=None
        self.ghsom_params=None
        self.ghsom_structure_graph = {}
        self.ghsom_nodes_by_coord_dict = {}


    def set_dataset(self,dataset,columns_names):
        self.dataset = np.copy(dataset)
        self.n_samples, self.n_features=dataset.shape
        self.columns_names = columns_names

    def get_dataset(self):
        return self.dataset

    def get_dataset_columns_names(self):
        return self.columns_names

    def get_dataset_atrib_names(self):
        return self.columns_names[0:len(self.columns_names)-1]

    
    def get_dataset_atrib_names_dcc_dropdown_format(self):

        atribs=  self.get_dataset_atrib_names()
        options = []  # must be a list of dicts per option

        for n in atribs:
            options.append({'label' : n, 'value': n})

        return options

        
    def get_data(self):
        return self.dataset[:,:-1]

    
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





    def set_gsom_model_info_dict(self,tam_eje_vertical,tam_eje_horizontal,tau_1,learning_rate,decadency,sigma,epocas_gsom,max_iter_gsom, check_semilla,seed):

        #Only one model could be used at once
        self.som_params= None
        self.gsom_params={}
        self.ghsom_params=None

        self.gsom_params['tam_eje_vertical'] = tam_eje_vertical
        self.gsom_params['tam_eje_horizontal'] =tam_eje_horizontal
        self.gsom_params['tau_1'] = tau_1
        self.gsom_params['learning_rate'] = learning_rate
        self.gsom_params['decadency'] = decadency
        self.gsom_params['sigma'] = sigma
        self.gsom_params['epocas_gsom'] = epocas_gsom
        self.gsom_params['max_iter_gsom'] = max_iter_gsom
        self.gsom_params['check_semilla'] = check_semilla
        self.gsom_params['seed'] = seed


    def set_gsom_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= None
        self.gsom_params = dict.copy()
        self.ghsom_params=None
            

    def get_gsom_model_info_dict(self):
        return  self.gsom_params







    def set_ghsom_model_info_dict(self,tau_1,tau_2,learning_rate,decadency,sigma,epocas_ghsom,max_iter_ghsom,check_semilla,seed):

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
        self.ghsom_params['check_semilla'] = check_semilla
        self.ghsom_params['seed'] = seed



    def set_ghsom_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= None
        self.gsom_params = None
        self.ghsom_params=dict.copy()
            

    def get_ghsom_model_info_dict(self):
        return  self.ghsom_params

    
    def get_ghsom_structure_graph(self):
        return self.ghsom_structure_graph

    def set_ghsom_structure_graph(self,graph):
        self.ghsom_structure_graph = graph 
    

    def get_ghsom_nodes_by_coord_dict(self):
        return self.ghsom_nodes_by_coord_dict 

    def set_ghsom_nodes_by_coord_dict(self,dict):
        self.ghsom_nodes_by_coord_dict = dict



            

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
    def save_model(model,filename):
        with open('filename', 'wb') as outfile:
            pickle.dump(model, outfile)


    @staticmethod
    def load_model(filename):
        with open('filename', 'rb') as infile:
            model = pickle.load(infile)
        return model


session_data = Sesion()