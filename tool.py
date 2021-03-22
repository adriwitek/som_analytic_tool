#This file loads  different apps on different urls 
'''
THIS IS THE FILE THAT KINKS UP ALL THE INDIVIDUAL PAGES 
It is worth noting that in both of these project structures, the Dash instance is defined in a separate app.py, while the entry point for running the app is index.py. 
This separation is required to avoid circular imports: the files containing the callback definitions require access to the Dash app instance however if this were
imported from index.py, the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''

import os

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from views.app import app
from views import callbacks
from views.home import Home
from views.training_selection import Training_selection
from views.train_som import train_som_view
from views.analyze_som_data import analyze_som_data
from views.analyze_gsom_data import analyze_gsom_data
from views.analyze_ghsom_data import analyze_ghsom_data
from views.session_data import *

from views import train_ghsom,train_gsom

# Dynamic Layout
def get_layout():
    return  html.Div([
                dcc.Location(id='url', refresh=False),
                html.Div(id='page-content')
            ])

app.layout = get_layout


#
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        #raise PreventUpdate
        return Home()
    elif pathname == URLS['TRAINING_SELECTION_URL']:
        return Training_selection()
    elif pathname == URLS['TRAINING_GHSOM_URL']:
        return train_ghsom.layout
    elif pathname == URLS['TRAINING_GSOM_URL']:
        return train_gsom.layout
    elif pathname == URLS['TRAINING_SOM_URL']:
        return train_som_view()
    elif pathname == URLS['ANALYZE_SOM_URL']:
        return analyze_som_data()
    elif pathname == URLS['ANALYZE_GSOM_URL']:
        return analyze_gsom_data()
    elif pathname == URLS['ANALYZE_GHSOM_URL']:
        return analyze_ghsom_data()
    else:
        return '404'



if __name__ == '__main__':
    if os.path.exists(SESSION_DATA_FILE_DIR) :
        os.remove(SESSION_DATA_FILE_DIR)
    app.run_server(debug=True)