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

import pandas as pd

from views.app import app
from views.elements import model_selector
from views.train_ghsom import layout as layout_train_ghsom

#from models.ghsom import GHSOM,GSOM
import numpy as np
import json

import  plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from views.custom_annotated_heatmap import create_annotated_heatmap as custom_heatmap

from  views.session_data import Sesion
from  config.config import *

 #TESTIN DATA
#PARA NO TENER QUE RECARGAR EL DATASET EN LAS PRUEBAS
data_to_plot = [[np.nan ,np.nan ,np.nan, np.nan, np.nan ,np.nan ,np.nan, 0],
                [np.nan ,np.nan, np.nan ,5 ,np.nan, np.nan ,np.nan ,np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [8 ,np.nan ,np.nan ,np.nan, np.nan, np.nan ,np.nan ,np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ,np.nan],
                [np.nan ,np.nan, 0, np.nan, np.nan, np.nan, np.nan, 0]]

data_to_plot = [[None ,None ,None, None, None ,None ,None, 0],
                [None ,None, None ,5 ,None, None ,None ,None],
                [None ,None, None ,None, None, None, None, None],
                [None ,None, None ,None, None, None, None, None],
                [8 ,None ,None ,None, None, None ,None ,None],
                [None ,None, None ,None, None, None, None, None],
                [None ,None, None, None, None, None, None ,None],
                [None ,None, 0, None, None, None, None, 0]]
                    



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











###################  TRAINING_SELECTION.PY CALLBACKS ######################

# TODO BORRAR ESTA FUNCION
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









   



@app.callback(Output('winners_map', 'figure'),
              Input('ver', 'n_clicks'),
              prevent_initial_call=True )
def update_som_fig(n_clicks):

    print('\nVISUALIZACION clicked\n')

    #Plasmamos datos en el json
    with open(SESSION_DATA_FILE_DIR) as json_file:
        session_data = json.load(json_file)

    tam_eje_x = session_data['som_tam_eje_x'] 
    tam_eje_y = session_data['som_tam_eje_y'] 

    #TODO : cambiar esto por guardado bien del dataset
    som = Sesion.modelo
    dataset = Sesion.data
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

   
    
    #print('targets',[t for t in targets])
    targets_list = [t[0] for t in targets.tolist()]
    #print('targetssss',targets_list)
    labels_map = som.labels_map(data, targets_list)
    data_to_plot = np.empty([tam_eje_x ,tam_eje_y],dtype=object)
    #data_to_plot[:] = np.nan#labeled heatmap does not support nonetypes

    if(session_data['discrete_data'] ):
        #showing the class more represented in each neuron
        for position in labels_map.keys():
            label_fracs = [ labels_map[position][t] for t in targets_list]
            max_value= max(label_fracs)
            winner_class_index = label_fracs.index(max_value)
            data_to_plot[position[0]][position[1]] = targets_list[winner_class_index]
    else: #continuos data: mean of the mapped values in each neuron
        for position in labels_map.keys():
            #fractions
            label_fracs = [ labels_map[position][t] for t in targets_list]
            data_to_plot[position[0]][position[1]] = np.mean(label_fracs)

    

   

   
    fig = go.Figure(data=go.Heatmap(
                       z=data_to_plot,
                       x=np.arange(tam_eje_x),
                       y=np.arange(tam_eje_y),
                       hoverongaps = True,
                       colorscale='Viridis'))
    fig.update_xaxes(side="top")
    '''


    x_ticks = np.linspace(0, tam_eje_x,tam_eje_x, dtype= int,endpoint=False).tolist()
    y_ticks = np.linspace(0, tam_eje_y,tam_eje_y,dtype= int, endpoint=False ).tolist()

    ######################################
    # ANNOTATED HEATMAPD LENTO
    #colorscale=[[np.nan, 'rgb(255,255,255)']]
    #fig = ff.create_annotated_heatmap(
    '''
    '''
    fig = custom_heatmap(
        #x= x_ticks,
        #y= y_ticks,
        z=data_to_plot,
        zmin=np.nanmin(data_to_plot),
        zmax=np.nanmax(data_to_plot),
        #xgap=5,
        #ygap=5,
        colorscale='Viridis',
        #colorscale=colorscale,
        #font_colors=font_colors,
        
        showscale=True #leyenda de colores
        )
    fig.update_layout(title_text='Clases ganadoras por neurona')
    fig['layout'].update(plot_bgcolor='white')
    '''

    
    #########################################################
    # ANNOTATED HEATMAPD RAPIDOOO
    
    #type= heatmap para mas precision
    #heatmapgl
    trace = dict(type='heatmap', z=data_to_plot, colorscale = 'Jet')
    data=[trace]

    # Here's the key part - Scattergl text! 
    


    data.append({'type': 'scattergl',
                    'mode': 'text',
                    #'x': x_ticks,
                    #'y': y_ticks,
                    'text': 'a'
                    })
    
    layout = {}
    layout['xaxis'] = {'range': [-0.5, tam_eje_x]}
    layout['width'] = 700
    layout['height']= 700
    annotations = []

    fig = dict(data=data, layout=layout)

    #condition_Nones = not(val is None)
    #condition_nans= not(np.isnan(val))



    #EIQUETANDO EL HEATMAP(solo los datos discretos)
    #Improved vers. for quick annotations by me
    if(session_data['discrete_data'] ):
        print('Etiquetando....')
        for n, row in enumerate(data_to_plot):
            for m, val in enumerate(row):
                 #font_color = min_text_color if ( val < self.zmid ) else max_text_color    esto lo haria aun mas lento
                if( not(val is None) ):
                    annotations.append(
                        go.layout.Annotation(
                           text= str(val) ,
                           x=m,
                           y=n,
                           #xref="x1",
                           #yref="y1",
                           #font=dict(color=font_color),
                           showarrow=False,
                        )
                    )
        

    
    
    layout['annotations'] = annotations
    

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

