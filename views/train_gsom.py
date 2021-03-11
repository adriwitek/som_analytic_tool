import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements

from models.ghsom import GSOM





# Formulario SOM
formulario_gsom =  dbc.ListGroupItem([
                    html.H4('Elección de parámetros',className="card-title"  )

                 

                ])









###############################   LAYOUT     ##############################
layout = html.Div(children=[

    elements.navigation_bar,
    elements.model_selector,
    formulario_gsom,
])


