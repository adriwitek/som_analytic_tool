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
import base64
import io

from views.app import app
from views.session_data import Sesion
from   views.elements import model_selector
from views.train_ghsom import layout as layout_train_ghsom

from models.ghsom import GHSOM,GSOM

sesion = None

#############################################################
###################  MAIN.PY CALLBACKS ######################
#############################################################
'''
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
@app.callback(Output('selector_modelo', 'label'),
              Input('seleccion_modelo_som','n_clicks'),
              Input('seleccion_modelo_gsom','n_clicks'),
              Input('seleccion_modelo_ghsom','n_clicks'),
              prevent_initial_call=True )
def dropdown_update_training_selection(n1,n2,n3):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (trigger_id == "seleccion_modelo_som"):
        return 'SOM'
    elif(trigger_id == "seleccion_modelo_gsom"):
        return 'GSOM'
    else:
        return 'GHSOM'




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
                sesion = Sesion(data)
            #elif 'xls' in filename:
                # Assume that the user uploaded an excel file
            #    df = pd.read_excel(io.BytesIO(decoded))

        except Exception as e:
            print(e)
            return html.Div([ 'There was an error processing this file.'])
        

        data = data.to_numpy()
        n_samples, n_features=data.shape
        cadena_1 = 'Número de datos: ' + str(n_samples)
        cadena_2 =  'Número de Atributos: ' + str(n_features - 1)
        #elements.session_data['n_samples'] = n_samples
        #elements.session_data['n_features'] = n_features

        # Give a default data dict with 0 clicks if there's no data.
        session_data = {}

        session_data['n_samples'] = n_samples
        session_data['n_features'] = n_features
        
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


####################################################
                    #SOM
####################################################













####################################################
                    #GHSOM
####################################################

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
def update_output(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana):
    '''Carga el dataset en los elementos adecuados

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

    dataset = sesion.data
    '''
    sesion.set_modelo(ghsom)
    
    ghsom = GHSOM(dataset , tau1, tau2, tasa_aprendizaje, decadencia, sigma_gaussiana)
    zero_unit = ghsom.train(epochs_number=15, dataset_percentage=1, min_dataset_size=1, seed=0, grow_maxiter=100)
    interactive_plot_with_labels(zero_unit.child_map, data, labels)
    plt.show()
    '''
    return 'entrenamiento_completado'

