# -*- coding: utf-8 -*-

'''
    Class session data file  
    
    Modelo para controlar las variables de estado de la aplicación grafica,
    con el patrón MVC.
'''
from logging import raiseExceptions
import numpy as np
import pickle

from  config.config import *
import time

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Sesion():
    
    
  

    def __init__(self):


        #Dataset
        self.discrete_data = True #TODO BORRAR ESTO
        self.file_data = None
     
    
        #DATA
            #if data not splitted, only train atribs will be used

        #   features
        self.features_names = []  #keep same order for ef. reasons
        self.features_dtypes = {}
        self.data_splitted = False
        self.df_features_train = None
        self.df_features_test = None
        self.np_features_train = None
        self.np_features_test = None
        #   targets
        self.target_name = None
        self.df_targets_train = None
        self.df_targets_test = None
        #   joined data,only saved if it's used 
        self.joined_train_test_np_data = None
        self.joined_train_test_df_targets = None




        #self.data= None     # numpy array
        #self.data_std= None # numpy array
        #self.targets_col =None# numpy array

    

        #tipo de modelo
        self.modelo = None
        self.som_params= None
        self.gsom_params=None
        self.ghsom_params=None
        self.ghsom_structure_graph = {}
        self.ghsom_nodes_by_coord_dict = {}

        #Progress bar ghsom porcentaje
        self.start_time = 0.0
        self.progressbar_maxvalue = 100
        self.progressbar_value = 0

        #Porgress bar gsom
        self.gsom_train_condition = - np.inf
        self.pbar_gsom_distancia_maxima = np.inf

        return


    #Call when closing app or loading home
    def clean_session_data(self):


        #Dataset
        self.discrete_data = True
        self.file_data = None
    
     
        #DATA
        #   features
        self.features_names = []  #keep same order for ef. reasons
        self.features_dtypes = {}
        self.data_splitted = False
        self.df_features_train = None
        self.df_features_test = None
        self.np_features_train = None
        self.np_features_test = None
        #   targets
        self.target_name = None
        self.df_targets_train = None
        self.df_targets_test = None
        #   joined data,only saved if it's used 
        self.joined_train_test_np_data = None
        self.joined_train_test_df_targets = None



        #todo check si todo ok

        #tipo de modelo
        self.modelo = None
        self.som_params= None
        self.gsom_params=None
        self.ghsom_params=None
        self.ghsom_structure_graph = {}
        self.ghsom_nodes_by_coord_dict = {}



    
    #estandariza y converte datos a numpy
    @staticmethod
    def estandarizar_data( df, string_info = '', data_splitted = False):
        if(not data_splitted):
            print('\t -->Standardizing ' + 'All' + ' Data...')
        else:
            print('\t -->Standardizing ' + string_info + ' Data...')
        scaler = preprocessing.StandardScaler()
        print('\t -->Standardizing Complete.')
        return  scaler.fit_transform(df)




    def set_target_name(self,target_name):    
        self.target_name = target_name

    def get_target_name(self):
        return self.target_name

    '''
    def get_target_np_column(self):
        #IF target selected
        if(self.get_target_name() is not None):
            self.targets_col
        else:
            return None
    '''

    

    def set_pd_dataframes(self,df_features,df_targets, split = False, train_samples_number=1.0 ):


        #self.n_samples, self.n_features = df_features.shape 

        self.features_names= df_features.columns.to_list()
        self.features_dtypes = df_features.dtypes.to_dict()
        self.data_splitted = split

        if(split):
            self.df_features_train , self.df_features_test, self.df_targets_train, self.df_targets_test = train_test_split(df_features, df_targets, train_size=train_samples_number ,shuffle = True)
        else:
            self.df_features_train= df_features
            self.df_targets_train = df_targets
            self.df_features_test= None
            self.df_targets_test= None
            self.joined_train_test_np_data= None
            self.joined_train_test_df_targets = None



    def convert_train_data_tonumpy(self):
        self.np_features_train = self.estandarizar_data(self.df_features_train, 'Train', self.data_splitted)


    def convert_test_data_tonumpy(self):
        if(self.data_splitted):
            self.np_features_test = self.estandarizar_data(self.df_features_test, 'Test',self.data_splitted )



    def get_train_data(self):

        if(self.np_features_train is None):
            self.convert_train_data_tonumpy()

        return self.np_features_train


    def get_test_data(self):

        if(not self.data_splitted):
            return self.get_train_data()
        elif(self.np_features_test is None):
            self.convert_test_data_tonumpy()

        return self.np_features_test



    def _create_joined_data(self):

        if(self.data_splitted):

            train_data = self.get_train_data()
            test_data = self.get_test_data()
            
            self.joined_train_test_np_data = np.concatenate((train_data, test_data), axis=0)
            frames = [self.df_targets_train,self.df_targets_test ]
            self.joined_train_test_df_targets =pd.concat(frames) 

            '''
            print('--DEBUG train test data\n' )
            print('train_data\n',train_data)
            print('test_data',test_data)
            print('joined_train_test_np_data\n',self.joined_train_test_np_data)
            print('-------')
            print('train_targe\n',self.df_targets_train)
            print('test_target\n',self.df_targets_test)
            print('joined_targets\n',self.joined_train_test_df_targets)
            '''



    def get_joined_train_test_np_data(self):

        if(not self.data_splitted):
            return self.get_train_data()

        elif(self.joined_train_test_np_data is None):
            self._create_joined_data()
        
        return self.joined_train_test_np_data 

    def get_joined_train_test_df_targets(self):

        if(not self.data_splitted):
            return self.df_targets_train

        elif(self.joined_train_test_df_targets is None):
            self._create_joined_data()
        
        return self.joined_train_test_df_targets 


    def get_data(self,option):

        '''
            option = 1 --> Train Data
            option = 2 --> Test Data
            option = 3 --> Train + Test Data
        '''

        if(option == 1 or not self.data_splitted):
            return self.get_train_data()
        elif(option == 2):
            return self.get_test_data()
        else:
            return self.get_joined_train_test_np_data()





    def get_features_names(self):
        return self.features_names


   
    def get_train_data_n_samples(self):
        n_samples, _ = self.get_train_data().shape
        return n_samples

    '''
    def get_data_n_features(self):
        return self.n_features
    '''
    
    def get_data_features_names_dcc_dropdown_format(self,columns= None):

        if(columns is None):
            atribs=  self.get_features_names()

        else:
            atribs= columns  

        options = []  # must be a list of dicts per option

        for n in atribs:
            options.append({'label' : n, 'value': n})

        return options






          
    def get_targets_list(self, option):

        '''
            option = 1 --> Train Data
            option = 2 --> Test Data
            option = 3 --> Train + Test Data
        '''


        if(option == 1 or  not self.data_splitted ):
            if(self.df_targets_train is not None):
                return self.df_targets_train[self.get_target_name()].tolist()
            else:
                return None
        elif(option == 2):
            if(self.df_targets_test is not None):
                return self.df_targets_test[self.get_target_name()].tolist()
            else:
                return None
        else:
            if(self.df_targets_test is not None and self.df_targets_train is not None):
                return self.get_joined_train_test_df_targets()[self.get_target_name()].tolist()
            else:
                return None




    def get_targets_options_dcc_dropdown_format(self):

        if(self.df_targets_train is not None):

            atribs = self.df_targets_train.columns

        else:
            atribs = []

        options = []  # must be a list of dicts per option

        for n in atribs:
            options.append({'label' : n, 'value': n})

        return options
        


    #dont call fun if not selected target
    def get_is_selected_target_numerical(self, option):

        
        '''
            option = 1 --> Train Data
            option = 2 --> Test Data
            option = 3 --> Train + Test Data
        '''
        

        if(option == 1 or  not self.data_splitted ):
            if(self.df_targets_train is not None):
                t_column =  self.df_targets_train[self.get_target_name()]
            else:
                t_column =  None
        elif(option == 2):
            if(self.df_targets_test is not None):
                t_column =  self.df_targets_test[self.get_target_name()]
            else:
                t_column =  None
        else:
            if(self.df_targets_test is not None and self.df_targets_train is not None):
                t_column =  self.get_joined_train_test_df_targets()[self.get_target_name()]
            else:
                t_column =  None

        if(t_column is None):
            raiseExceptions('Unexpedted error')
        elif( pd.api.types.is_string_dtype(t_column) or pd.api.types.is_bool_dtype(t_column)):
            #print('es string o bool el target')
            return False
        else:
            #print('el target es numerico')
            return True



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
    

    #TODO BORRAR SI NO LA USO
    def get_current_model_type(self):

        if(self.som_params is not None):
            return 'SOM'
        elif(self.gsom_params is not None):
            return 'GSOM'
        elif(self.ghsom_params is not None):
            return 'GHSOM'
        else:
            ''

    #TODO BORRAR SI NO LA USO
    def get_current_model_type_analyze_url(self):

        if(self.som_params is not None):
            return URLS['ANALYZE_SOM_URL']
        elif(self.gsom_params is not None):
            return URLS['ANALYZE_GSOM_URL']
        elif(self.ghsom_params is not None):
            return URLS['ANALYZE_GHSOM_URL']
        else:
            ''

  
        


    def set_som_model_info_dict(self,tam_eje_vertical,tam_eje_horizontal,learning_rate,neigh_fun,distance_fun,
                                sigma,iteraciones, inicialitacion_pesos,check_semilla, semilla):

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
        self.som_params['check_semilla'] = check_semilla
        self.som_params['seed'] = semilla

            
    def set_som_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= dict.copy()
        self.gsom_params = None
        self.ghsom_params=None

    def get_som_model_info_dict(self):
        return  self.som_params





    def set_gsom_model_info_dict(self,tam_eje_vertical,tam_eje_horizontal,tau_1,learning_rate,decadency,sigma,epocas_gsom,max_iter_gsom,
                                fun_disimilitud, check_semilla,seed):

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
        self.gsom_params['fun_disimilitud'] = fun_disimilitud
        self.gsom_params['check_semilla'] = check_semilla
        self.gsom_params['seed'] = seed


    def set_gsom_model_info_dict_direct(self,dict):
        #for laod model purpose
        self.som_params= None
        self.gsom_params = dict.copy()
        self.ghsom_params=None
            

    def get_gsom_model_info_dict(self):
        return  self.gsom_params







    def set_ghsom_model_info_dict(self,tau_1,tau_2,learning_rate,decadency,sigma,epocas_ghsom,max_iter_ghsom,
                                    fun_disimilitud,check_semilla,seed):

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
        self.ghsom_params['fun_disimilitud'] = fun_disimilitud
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


    @staticmethod
    def save_model(model,filename):
        with open('filename', 'wb') as outfile:
            pickle.dump(model, outfile)


    @staticmethod
    def load_model(filename):
        with open('filename', 'rb') as infile:
            model = pickle.load(infile)
        return model

    def get_features_dtypes(self):
        return self.features_dtypes




    #GHSOM PROGRESS BAR FUNS
    def set_progressbar_maxvalue(self,maxvalue):
        self.progressbar_maxvalue = maxvalue

    def get_progressbar_maxvalue(self):
        return self.progressbar_maxvalue

    def update_progressbar_value(self,valor):
        porcentaje = (valor*100)/self.progressbar_maxvalue
        self.progressbar_value = porcentaje

    def get_progressbar_value(self):
        return self.progressbar_value

    def reset_progressbar_value(self):
        self.progressbar_value = 0


    #GSOM PROGRESS BAR FUNS
    def set_pbar_gsom_train_condition(self,gsom_train_condition):
        #gsom_train_condition = tau1*qe0
        self.gsom_train_condition = gsom_train_condition

    def get_pbar_gsom_train_condition(self):
        #gsom_train_condition = tau1*qe0
        return self.gsom_train_condition 

    def set_pbar_gsom_distancia_maxima(self,distancia_maxima):
        self.pbar_gsom_distancia_maxima = distancia_maxima

    def update_gsom_progressbar_value(self,distancia):
        '''Se calcula el progreso como distancia del MQE del gsom a la condicion de parada.

            distanca = MQEmapa - (tau1 - qe0)

        Progreso(%)    
                    
                100%    |
                        |\
                        | \ Pendiente
                        |  \
                0%      |___\__________     <--distancia_maxima       
                                Distancia


        '''

        if(distancia<=0):
            self.progressbar_value = 100
        elif(distancia >= self.pbar_gsom_distancia_maxima):
            self.progressbar_value = 0
        else:
            self.progressbar_value = (distancia * (-100/self.pbar_gsom_distancia_maxima) ) +100

 

    def start_timer(self):
        self.start_time= time.time()

    def get_training_elapsed_time(self):
        now = time.time()
        return now - self.start_time

    

session_data = Sesion()