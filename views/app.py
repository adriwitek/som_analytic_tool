import dash
import dash_bootstrap_components as dbc

'''
the files containing the callback definitions require access to the Dash app instance however if this were imported from index.py,
the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''

app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.LITERA])
#app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LITERA])

server = app.server