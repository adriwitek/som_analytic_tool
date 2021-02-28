import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements

import  plotly.express as px

fig = px.scatter()



def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        dcc.Graph(
            id='winners_map',
            figure=fig
        ),

        html.Div( 
            [dbc.Button("Ver", id="ver", className="mr-2", color="primary")],
            style={'textAlign': 'center'}
        ),




    ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        body,
    ])


    return layout
