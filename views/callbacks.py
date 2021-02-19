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

from GHSOM import GHSOM
from views.app import app
from views.session_data import Sesion
import  views.elements as elements


sesion = None

#############################################################
###################  MAIN.PY CALLBACKS ######################
#############################################################

@app.callback(Output('hidden_div_for_redirect_callback', 'children'),
              Input('continue-button', 'n_clicks'), prevent_initial_call=True )
def redirect_to_training_selection(n_clicks):
    return dcc.Location(pathname="/training_selection", id="redirect")


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



@app.callback(Output('table_info_n_samples', 'children'),
              Output('table_info_n_features', 'children'),
              Input('session_data','modified_timestamp'),
              State('session_data', 'data'),
              prevent_initial_call=False)
def update_dataset_info_table( data, session_data):
    return session_data['n_samples'] ,session_data['n_features']

@app.callback(Output('ghsom_tree_structure', 'figure'),
              Input('train-button', 'n_clicks'),
              State('tau1', 'value'),
              State('tau2', 'value'),
              State('tasa_aprendizaje','value'),
              State('decadencia','value'),
              State('sigma','value'), 
              prevent_initial_call=True)
def train_data_with_ghsom(n_clicks, tau1, tau2, tasa_aprendizaje, decadencia, sigma ):
    
    #ghsom = GHSOM(input_dataset=sesion.data , t1=0.1, t2=0.0001, learning_rate=0.15, decay=0.95, gaussian_sigma=1.5)
    ghsom = GHSOM(input_dataset=sesion.data , t1=tau1, t2=tau2, learning_rate=tasa_aprendizaje, decay=decadencia, gaussian_sigma=sigma)
    sesion.set_modelo(ghsom)

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




