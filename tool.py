#This file loads  different apps on different urls 
'''
It is worth noting that in both of these project structures, the Dash instance is defined in a separate app.py, while the entry point for running the app is index.py. 
This separation is required to avoid circular imports: the files containing the callback definitions require access to the Dash app instance however if this were
imported from index.py, the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from views.app import app
from views import callbacks
from views import main,training_selection



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        #raise PreventUpdate
        return main.layout
    elif pathname == '/training_selection':
       return training_selection.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)