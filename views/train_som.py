import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements



def train_som_view():

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([
                html.H4('Elección de parámetros',className="card-title"  ),
                html.Div(
                        #style={'textAlign': 'center'}
                        children=[

                            html.H5(children='Tamaño del grid(eje X):'),
                            dcc.Input(id="tam_eje_x", type="number", value=8,step=1,min=1),

                            html.H5(children='Tamaño del grid(eje Y):'),
                            dcc.Input(id="tam_eje_y", type="number", value=8,step=1,min=1),

                            html.H5(children='Tasa de aprendizaje:'),
                            dcc.Input(id="tasa_aprendizaje_som", type="number", value="0.5",step=0.01,min=0,max=5),


                            html.H5(children='Función de vecindad'),
                            dcc.Dropdown(
                                id='dropdown_vecindad',
                                options=[
                                    {'label': 'Gaussiana', 'value': 'gaussian'},
                                    {'label': 'Sombrero Mejicano', 'value': 'mexican_hat'},
                                    {'label': 'Burbuja', 'value': 'bubble'},
                                    {'label': 'Triángulo', 'value': 'triangle'}
                                ],
                                value='gaussian',
                                searchable=False,
                                style={'width': '35%'}
                            ),


                            html.H5(children='Topologia del mapa'),
                            dcc.Dropdown(
                                id='dropdown_topology',
                                options=[
                                    {'label': 'Rectangular', 'value': 'rectangular'},
                                    {'label': 'Hexagonal', 'value': 'hexagonal'}
                                ],
                                value='rectangular',
                                searchable=False,
                                style={'width': '35%'}
                            ),


                            html.H5(children='Función de distancia'),
                            dcc.Dropdown(
                                id='dropdown_distance',
                                options=[
                                    {'label': 'Euclidea', 'value': 'euclidean'},
                                    {'label': 'Coseno', 'value': 'cosine'},
                                    {'label': 'Manhattan', 'value': 'manhattan'},
                                    {'label': 'Chebyshev', 'value': 'chebyshev'}
                                ],
                                value='euclidean',
                                searchable=False,
                                style={'width': '35%'}
                            ),


                            html.H5(children='Sigma gaussiana:'),
                            dcc.Input(id="sigma", type="number", value="1.5",step=0.01,min=0,max=10),


                            html.H5(children='Iteracciones:'),
                            dcc.Input(id="iteracciones", type="number", value="1000",step=1,min=1),

                            html.H5(children='Inicialización pesos del mapa'),
                            dcc.Dropdown(
                                id='dropdown_inicializacion_pesos',
                                options=[
                                    {'label': 'PCA: Análisis de Componentes Principales ', 'value': 'pca'},
                                    {'label': 'Aleatoria', 'value': 'random'},
                                    {'label': 'Sin inicialización de pesos', 'value': 'no_init'}
                                ],
                                value='pca',
                                searchable=False,
                                style={'width': '45%'}
                            ),
                            html.Hr(),

                            html.Div( 
                                [dbc.Button("Entrenar", id="train_button_som",href='analyze-som-data',disabled= True, className="mr-2", color="primary")],
                                style={'textAlign': 'center'}
                            ),
                            html.H6(id='som_entrenado')

                ])
            ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        elements.model_selector,
        formulario_som,
    ])


    return layout




