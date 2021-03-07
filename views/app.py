import dash
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.LITERA])
#app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LITERA])

server = app.server