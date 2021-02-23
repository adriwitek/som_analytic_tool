import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements



def train_som_view():

    # Formulario SOM
    formulario_som =  dbc.ListGroupItem([
                        html.H4('Elección de parámetros',className="card-title"  ),

                        html.H5(children='Tamaño del grid(eje X):'),
                        dcc.Input(id="tam_eje_x", type="number", value=2,step=1,min=1),

                        html.H5(children='Tamaño del grid(eje Y):'),
                        dcc.Input(id="tam_eje_y", type="number", value=2,step=1,min=1),

                        html.H5(children='Tasa de aprendizaje:'),
                        dcc.Input(id="tasa_aprendizaje_som", type="number", value="0.5",step=0.01,min=0,max=5),


                        html.H5(children='Función de vecindad'),
                        dbc.DropdownMenu(
                            label="Gaussiana",
                            id= 'dropdown_sigma_gaussiana_som',
                            children=[
                                dbc.DropdownMenuItem("Gaussiana"),
                                dbc.DropdownMenuItem("Sombrero Mejicano"),
                                dbc.DropdownMenuItem("Burbuja"),
                                dbc.DropdownMenuItem("Triángulo"),
                            ],
                        ),

                        html.H5(children='Topologia del mapa'),
                        dbc.DropdownMenu(
                            label="Rectangular",
                            children=[
                                dbc.DropdownMenuItem("Rectangular"),
                                dbc.DropdownMenuItem("Hexagonal"),
                            ],
                        ),


                        html.H5(children='Función de distancia'),
                        dbc.DropdownMenu(
                            label="Euclidea",
                            children=[
                                dbc.DropdownMenuItem("Euclidea"),
                                dbc.DropdownMenuItem("Coseno"),
                                dbc.DropdownMenuItem("Manhattan"),
                                dbc.DropdownMenuItem("Chebyshev"),
                            ],
                        ),


                        html.H5(children='Sigma gaussiana:'),
                        dcc.Input(id="sigma", type="number", value="1.5",step=0.01,min=0,max=10),
                        html.Hr(),

                        html.Div( 
                            [dbc.Button("Entrenar", id="train_button_som",disabled= True, className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                        html.P(id='test_element')

                    ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        elements.model_selector,
        formulario_som,
    ])


    return layout




