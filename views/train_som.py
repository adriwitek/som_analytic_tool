
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements
import json
from models.som import minisom


from views.session_data import Sesion

def train_som_view():

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([
                html.H4('Elección de parámetros',className="card-title"  ),
                html.Div(
                        #style={'textAlign': 'center'}
                        children=[

                            html.H5(children='Tamaño del grid(eje X):'),
                            dcc.Input(id="tam_eje_x", type="number", value=2,step=1,min=1),

                            html.H5(children='Tamaño del grid(eje Y):'),
                            dcc.Input(id="tam_eje_y", type="number", value=2,step=1,min=1),

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

                            html.Div(children=[
                                dbc.Button("Entrenar", id="train_button_som",href='analyze-som-data',disabled= True, className="mr-2", color="primary")]
                                #,dbc.Spinner(id='spinner_training',color="primary",fullscreen=False)],
                                #    style={'textAlign': 'center'}
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










#Habilitar boton train som
@app.callback(Output('train_button_som','disabled'),
              Input('tam_eje_x', 'value'),
              Input('tam_eje_y', 'value'),
              Input('tasa_aprendizaje_som', 'value'),
              Input('dropdown_vecindad', 'value'),
              Input('dropdown_topology', 'value'),
              Input('dropdown_distance', 'value'),
              Input('sigma', 'value'),
              Input('iteracciones', 'value'),
              Input('dropdown_inicializacion_pesos','value')
            )
def enable_train_som_button(tam_eje_x,tam_eje_y,tasa_aprendizaje,vecindad, topology, distance,
                            sigma,iteracciones,dropdown_inicializacion_pesos):
    if all(i is not None for i in [tam_eje_x,tam_eje_y,tasa_aprendizaje,vecindad, topology, distance,
                                    sigma,iteracciones,dropdown_inicializacion_pesos]):
        return False
    else:
        return True





@app.callback(Output('som_entrenado', 'children'),
              Input('train_button_som', 'n_clicks'),
              State('tam_eje_x', 'value'),
              State('tam_eje_y', 'value'),
              State('tasa_aprendizaje_som', 'value'),
              State('dropdown_vecindad', 'value'),
              State('dropdown_topology', 'value'),
              State('dropdown_distance', 'value'),
              State('sigma', 'value'),
              State('iteracciones', 'value'),
              State('dropdown_inicializacion_pesos','value'),
              prevent_initial_call=True )
def train_som(n_clicks,x,y,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,pesos_init):

    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)


    # TRAINING
    dataset = Sesion.data

    #Plasmamos datos en el json
    with open('data_session.json') as json_file:
        session_data = json.load(json_file)

    session_data['som_tam_eje_x'] = x
    session_data['som_tam_eje_y'] = y

    with open('data_session.json', 'w') as outfile:
        json.dump(session_data, outfile)


    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

    som = minisom.MiniSom(x=x, y=y, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                neighborhood_function=vecindad, topology=topology,
                 activation_distance=distance, random_seed=None)
    
    #Weigh init
    if(pesos_init == 'pca'):
        som.pca_weights_init(data)
    elif(pesos_init == 'random'):   
        som.random_weights_init(data)

    som.train(data, iteracciones, verbose=True)  # random training                                                          #quitar el verbose
    Sesion.modelo = som

    print('ENTRENAMIENTO FINALIZADO')

    return 'Entrenamiento completado',session_data

