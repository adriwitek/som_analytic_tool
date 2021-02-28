# -*- coding: utf-8 -*-

'''
the files containing the callback definitions require access to the Dash app instance however if this were imported from index.py,
the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html

from datetime import datetime
from pandas import read_csv
import pandas as pd
import base64
import io
from io import BytesIO
from views.app import app
from views.session_data import Sesion,session_data_dict
from views.elements import model_selector
from views.train_ghsom import layout as layout_train_ghsom

#from models.ghsom import GHSOM,GSOM
from models.som import minisom
import numpy as np
import json

import  plotly.express as px
import plotly.graph_objects as go



#############################################################
###################  MAIN.PY CALLBACKS ######################
#############################################################
'''
ESTO HACERLO PARA CUANDO EL DATASET ES NONE REDIRIGIR A HOME!!!!!!!!!!!!!!!!!!

@app.callback(Output('hidden_div_for_redirect_callback', 'children'),
              Input('continue-button', 'n_clicks'),
              prevent_initial_call=True )
def update_app_content_view(n1,n2,n3,n4):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (trigger_id == "continue-button"):
        return dcc.Location(pathname="/train-som", id="redirect")
    elif (trigger_id == "seleccion_modelo_som"):
            return dcc.Location(pathname="/train-som", id="redirect")
    elif(trigger_id == "seleccion_modelo_gsom"):
        return dcc.Location(pathname="/train-gsom", id="redirect")
    elif(trigger_id == "seleccion_modelo_gsom"):
        return dcc.Location(pathname="/train-ghsom", id="redirect")
    else:
        return dcc.Location(pathname="/", id="redirect")
'''

#Dropdown selector modelo
@app.callback(Output('label_selected_model', 'children'),
              Input('seleccion_modelo_som','n_clicks'),
              Input('seleccion_modelo_gsom','n_clicks'),
              Input('seleccion_modelo_ghsom','n_clicks'),
              Input('url', 'pathname') )
def dropdown_update_training_selection(n1,n2,n3,url):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]


    if (trigger_id == "seleccion_modelo_ghsom" or 'ghsom' in url ):
        return 'GHSOM'
    elif (trigger_id == "seleccion_modelo_gsom" or 'gsom' in url ):
        return 'GSOM'
    else:
        return 'SOM' 




@app.callback(Output('output-data-upload_1', 'children'),
              Output('output-data-upload_2', 'children'),
              Output('continue-button','disabled'),
              Output('n_samples','children'),
              Output('n_features','children'),
              Output('session_data','data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), 
              State('session_data', 'data'),
                prevent_initial_call=True)
def update_output(contents, filename, last_modified,session_data):
    '''Carga el dataset en los elementos adecuados

    '''
    if contents is not None:
        output_1= 'Archivo: ' + filename
        output_2= 'Última Modificación: ' +  datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S')

        #guarda_dataframe(contents)
        #Esta carga necesita ser asi
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                data = read_csv(io.StringIO(decoded.decode('utf-8')))
                

            #elif 'xls' in filename:
                # Assume that the user uploaded an excel file
            #    df = pd.read_excel(io.BytesIO(decoded))

        except Exception as e:
            print(e)
            return html.Div([ 'There was an error processing this file.'])
        

       

        data = data.to_numpy()
        n_samples, n_features=data.shape

        Sesion.data = data
        Sesion.n_samples, Sesion.n_features=  n_samples, n_features
        cadena_1 = 'Número de datos: ' + str(n_samples)
        cadena_2 =  'Número de Atributos: ' + str(n_features - 1)
 
        
        session_data = session_data_dict()
        session_data['n_samples'] = n_samples
        session_data['n_features'] = n_features
        with open('data_session.json', 'w') as outfile:
            json.dump(session_data, outfile)

        # Give a default data dict with 0 clicks if there's no data.
        
        
        
        return output_1, output_2,False,cadena_1,cadena_2, session_data
    else:
        return '','',True,'','','',{}







###################  TRAINING_SELECTION.PY CALLBACKS ######################


#Actualizar tabla info dataset
@app.callback(Output('table_info_n_samples', 'children'),
              Output('table_info_n_features', 'children'),
              Input('session_data','modified_timestamp'),
              State('session_data', 'data'),
              prevent_initial_call=False)
def update_dataset_info_table( data, session_data):
    return session_data['n_samples'] ,session_data['n_features']










############################################################################################################################################################
                    #SOM
############################################################################################################################################################


#Habilitar boton train som
@app.callback(Output('train_button_som','disabled'),
              Input('tam_eje_x', 'value'),
              Input('tam_eje_y', 'value'),
              Input('tasa_aprendizaje_som', 'value'),
              Input('dropdown_vecindad', 'value'),
              Input('dropdown_topology', 'value'),
              Input('dropdown_distance', 'value'),
              Input('sigma', 'value'))
def enable_train_som_button(tam_eje_x,tam_eje_y,tasa_aprendizaje,vecindad, topology, distance,sigma):

    if all(i is not None for i in [tam_eje_x,tam_eje_y,tasa_aprendizaje,vecindad, topology, distance,sigma]):
        return False
    else:
        return True




@app.callback(Output('som_entrenado', 'children'),
              Input('train_button_som', 'n_clicks'),
              State('tam_eje_x', 'value'),
              State('tam_eje_y', 'value'),
              State('tasa_aprendizaje_som', 'value'),
              State('dropdown_vecindad', 'value'),
              State('dropdown_topology', 'value'),
              State('dropdown_distance', 'value'),
              State('sigma', 'value'),
              prevent_initial_call=True )
def train_som(n_clicks,x,y,tasa_aprendizaje,vecindad, topology, distance,sigma):

    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)



    # TRAINING
    dataset = Sesion.data

    #Plasmamos datos en el json
    with open('data_session.json') as json_file:
        session_data = json.load(json_file)

    session_data['som_tam_eje_x'] = x
    session_data['som_tam_eje_y'] = y

    with open('data_session.json', 'w') as outfile:
        json.dump(session_data, outfile)


    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

    som = minisom.MiniSom(x=x, y=y, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                neighborhood_function=vecindad, topology=topology,
                 activation_distance=distance, random_seed=None)
    
    som.pca_weights_init(data)
    som.train(data, 1000, verbose=True)  # random training
    Sesion.modelo = som
    print('ENTRENAMIENTO FINALIZADO')

    return 'entrenadoooooo000000000000000000000oooooooo',session_data

   



@app.callback(Output('winners_map', 'figure'),
              Input('ver', 'n_clicks'),
              prevent_initial_call=True )
def update_som_fig(n_clicks):

    print('\nVISUALIZACION clicked\n')

    #Plasmamos datos en el json
    with open('data_session.json') as json_file:
        session_data = json.load(json_file)

    tam_eje_x = session_data['som_tam_eje_x'] 
    tam_eje_y = session_data['som_tam_eje_y'] 


    som = Sesion.modelo
    dataset = Sesion.data
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

    # VISUALIZACION   

   
    
    #print('targets',[t for t in targets])
    targets_list = [t[0] for t in targets.tolist()]
    #print('targetssss',targets_list)
    labels_map = som.labels_map(data, targets_list)
    data_to_plot = np.empty([tam_eje_x ,tam_eje_y],dtype=object)
    

    for position in labels_map.keys():
        label_fracs = [ labels_map[position][t] for t in targets_list]
        max_value= max(label_fracs)
        winner_class_index = label_fracs.index(max_value)
        data_to_plot[position[0]][position[1]] = targets_list[winner_class_index]

    #print(data_to_plot)

    fig = go.Figure(data=go.Heatmap(
                       z=data_to_plot,
                       x=np.arange(tam_eje_x),
                       y=np.arange(tam_eje_y),
                       hoverongaps = True,
                       colorscale='Viridis'))
   


    fig.update_xaxes(side="top")

    
    print('\nVISUALIZACION:renderfinalizado\n')

    return fig













########################################################################################################
                    #GHSOM
########################################################################################################


# Sync slider tau1
@app.callback(
    Output("tau1", "value"),
    Output("tau1_slider", "value"),
    Input("tau1", "value"),
    Input("tau1_slider", "value"), prevent_initial_call=True)
def sync_slider_tau1(tau1, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau1 if trigger_id == "tau1" else slider_value
    return value, value


# Sync slider tau2
@app.callback(
    Output("tau2", "value"),
    Output("tau2_slider", "value"),
    Input("tau2", "value"),
    Input("tau2_slider", "value"), prevent_initial_call=True)
def sync_slider_tau2(tau2, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau2 if trigger_id == "tau2" else slider_value
    return value, value


#Habilitar boton train ghsom
@app.callback(Output('train_button_ghsom','disabled'),
              Input('tau1','value'),
              Input('tau2','value'),
              Input('tasa_aprendizaje','value'),
              Input('decadencia','value'),
              Input('sigma','value'))
def enable_train_ghsom_button(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana):
    '''Habilita el boton de train del ghsom

    '''
    if all(i is not None for i in [tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana]):
        return False
    else:
        return True



#Boton train ghsom
@app.callback(Output('test_element', 'value'),
              Input('train_button_ghsom_button', 'n_clicks'),
              State('tau1','value'),
              State('tau2','value'),
              State('tasa_aprendizaje','value'),
              State('decadencia','value'),
              State('sigma','value'),
              prevent_initial_call=True )
def train_ghsom(n_clicks,tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana):

    #dataset = esion.data
    '''
    sesion.set_modelo(ghsom)
    
    ghsom = GHSOM(dataset , tau1, tau2, tasa_aprendizaje, decadencia, sigma_gaussiana)
    zero_unit = ghsom.train(epochs_number=15, dataset_percentage=1, min_dataset_size=1, seed=0, grow_maxiter=100)
    interactive_plot_with_labels(zero_unit.child_map, data, labels)
    plt.show()
    '''
    return 'entrenamiento_completado'

