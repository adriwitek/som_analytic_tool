
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
import numpy as np
import plotly.graph_objects as go


from  views.session_data import session_data
from  config.config import *

def train_som_view():

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([
                html.H4('Elección de parámetros',className="card-title"  ),
                html.Div(
                        #style={'textAlign': 'center'}
                        children=[

                            html.H5(children='Tamaño del grid(Eje vertical):'),
                            dcc.Input(id="tam_eje_vertical", type="number", value=2,step=1,min=1),

                            html.H5(children='Tamaño del grid(Eje horizontal):'),
                            dcc.Input(id="tam_eje_horizontal", type="number", value=2,step=1,min=1),

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
              Input('tam_eje_vertical', 'value'),
              Input('tam_eje_horizontal', 'value'),
              Input('tasa_aprendizaje_som', 'value'),
              Input('dropdown_vecindad', 'value'),
              Input('dropdown_topology', 'value'),
              Input('dropdown_distance', 'value'),
              Input('sigma', 'value'),
              Input('iteracciones', 'value'),
              Input('dropdown_inicializacion_pesos','value')
            )
def enable_train_som_button(tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                            sigma,iteracciones,dropdown_inicializacion_pesos):
    if all(i is not None for i in [tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                                    sigma,iteracciones,dropdown_inicializacion_pesos]):
        return False
    else:
        return True





@app.callback(Output('som_entrenado', 'children'),
              Input('train_button_som', 'n_clicks'),
              State('tam_eje_vertical', 'value'),
              State('tam_eje_horizontal', 'value'),
              State('tasa_aprendizaje_som', 'value'),
              State('dropdown_vecindad', 'value'),
              State('dropdown_topology', 'value'),
              State('dropdown_distance', 'value'),
              State('sigma', 'value'),
              State('iteracciones', 'value'),
              State('dropdown_inicializacion_pesos','value'),
              prevent_initial_call=True )
def train_som(n_clicks,eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,pesos_init):

    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)


    # TRAINING
    dataset = session_data.get_data()

    #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
    session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,iteracciones, pesos_init)

    #TODO BORRAR ESTO DEL JSON

    #Plasmamos datos en el json
 
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

    som = minisom.MiniSom(x=eje_vertical, y=eje_horizontal, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                neighborhood_function=vecindad, topology=topology,
                 activation_distance=distance, random_seed=None)
    
    #Weigh init
    if(pesos_init == 'pca'):
        som.pca_weights_init(data)
    elif(pesos_init == 'random'):   
        som.random_weights_init(data)

    som.train(data, iteracciones, verbose=True)  # random training   
    session_data.set_modelo(som)                                                       # TODO quitar el verbose

    print('ENTRENAMIENTO FINALIZADO')

    return 'Entrenamiento completado',session_data









@app.callback(Output('winners_map', 'figure'),
              Input('ver', 'n_clicks'),
              prevent_initial_call=True )
def update_som_fig(n_clicks):

    print('\nVISUALIZACION clicked\n')


    params = session_data.get_som_model_info_dict()
    #TODO borrar
    '''
    with open(SESSION_DATA_FILE_DIR) as json_file:
        datos_entrenamiento = json.load(json_file)

    tam_eje_vertical = datos_entrenamiento['som_tam_eje_vertical'] 
    tam_eje_horizontal = datos_entrenamiento['som_tam_eje_horizontal'] 
    '''
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']
    
    #TODO : cambiar esto por guardado bien del dataset

    som = session_data.get_modelo()
    dataset = session_data.get_data()
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

   
    
    #print('targets',[t for t in targets])
    targets_list = [t[0] for t in targets.tolist()]
    #print('targetssss',targets_list)
    labels_map = som.labels_map(data, targets_list)
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    #data_to_plot[:] = np.nan#labeled heatmap does not support nonetypes

    if(session_data.get_discrete_data() ):
        #showing the class more represented in each neuron
        for position in labels_map.keys():
            label_fracs = [ labels_map[position][t] for t in targets_list]
            max_value= max(label_fracs)
            winner_class_index = label_fracs.index(max_value)
            data_to_plot[position[0]][position[1]] = targets_list[winner_class_index]
    else: #continuos data: mean of the mapped values in each neuron
        for position in labels_map.keys():
            #fractions
            label_fracs = [ labels_map[position][t] for t in targets_list]
            data_to_plot[position[0]][position[1]] = np.mean(label_fracs)

    

   

   
    fig = go.Figure(data=go.Heatmap(
                       z=data_to_plot,
                       x=np.arange(tam_eje_vertical),
                       y=np.arange(tam_eje_horizontal),
                       hoverongaps = True,
                       colorscale='Viridis'))
    fig.update_xaxes(side="top")
    '''


    x_ticks = np.linspace(0, tam_eje_vertical,tam_eje_vertical, dtype= int,endpoint=False).tolist()
    y_ticks = np.linspace(0, tam_eje_horizontal,tam_eje_horizontal,dtype= int, endpoint=False ).tolist()

    ######################################
    # ANNOTATED HEATMAPD LENTO
    #colorscale=[[np.nan, 'rgb(255,255,255)']]
    #fig = ff.create_annotated_heatmap(
    '''
    '''
    fig = custom_heatmap(
        #x= x_ticks,
        #y= y_ticks,
        z=data_to_plot,
        zmin=np.nanmin(data_to_plot),
        zmax=np.nanmax(data_to_plot),
        #xgap=5,
        #ygap=5,
        colorscale='Viridis',
        #colorscale=colorscale,
        #font_colors=font_colors,
        
        showscale=True #leyenda de colores
        )
    fig.update_layout(title_text='Clases ganadoras por neurona')
    fig['layout'].update(plot_bgcolor='white')
    '''

    
    #########################################################
    # ANNOTATED HEATMAPD RAPIDOOO
    
    #type= heatmap para mas precision
    #heatmapgl
    trace = dict(type='heatmap', z=data_to_plot, colorscale = 'Jet')
    data=[trace]

    # Here's the key part - Scattergl text! 
    


    data.append({'type': 'scattergl',
                    'mode': 'text',
                    #'x': x_ticks,
                    #'y': y_ticks,
                    'text': 'a'
                    })
    
    layout = {}
    layout['xaxis'] = {'range': [-0.5, tam_eje_vertical]}
    layout['width'] = 700
    layout['height']= 700
    annotations = []

    fig = dict(data=data, layout=layout)

    #condition_Nones = not(val is None)
    #condition_nans= not(np.isnan(val))



    #EIQUETANDO EL HEATMAP(solo los datos discretos)
    #Improved vers. for quick annotations by me
    if(session_data.get_discrete_data() ):
        print('Etiquetando....')
        for n, row in enumerate(data_to_plot):
            for m, val in enumerate(row):
                 #font_color = min_text_color if ( val < self.zmid ) else max_text_color    esto lo haria aun mas lento
                if( not(val is None) ):
                    annotations.append(
                        go.layout.Annotation(
                           text= str(val) ,
                           x=m,
                           y=n,
                           #xref="x1",
                           #yref="y1",
                           #font=dict(color=font_color),
                           showarrow=False,
                        )
                    )
        

    
    
    layout['annotations'] = annotations
    

    print('\nVISUALIZACION:renderfinalizado\n')

    return fig


