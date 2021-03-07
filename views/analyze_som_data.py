import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from views.session_data import Sesion,session_data_dict
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil
import numpy as np

fig = go.Figure()




def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        html.Hr(),
        html.Div(children=[ 

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_1"))
                ),
                dbc.Collapse(id="collapse_1",children=
                    dbc.CardBody(children=[ 
                        html.Div([dcc.Graph(id='winners_map',figure=fig)],
                          style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),


            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Component plans",color="link",id="button_collapse_2"))
                ),
                dbc.Collapse(id="collapse_2",children=
                    dbc.CardBody(children=[
                        html.H5("Seleccionar atributos para mostar:"),
                        dcc.Dropdown(
                            id='dropdown_atrib_names',
                            options=get_nombres_atributos(),
                            multi=True
                        ),
                        html.Div( 
                            [dbc.Button("Ver Mapas de Componentes", id="ver_mapas_componentes_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                        html.Div([dcc.Graph(id='mapa_componentes_fig',figure=fig)],
                          style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        )

                    ]),
                ),
            ]),



            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("3333",color="link",id="button_collapse_3"))
                ),
                dbc.Collapse(id="collapse_3",children=
                    dbc.CardBody(
                        #METER AQUI LO QUE SEAA



                    ),
                ),
            ])



        ])

        
        
        
    
        
    



    ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        body,
    ])

    return layout







def get_nombres_atributos():

    with open('data_session.json') as json_file:
        session_data = json.load(json_file)

    nombres = session_data['columns_names']
    atribs= nombres[0:len(nombres)-1]
    options = []  # must be a list of dicts per option

    for n in atribs:
        options.append({'label' : n, 'value': n})

    return options





##################################################################
#                       CALLBACKS
##################################################################

@app.callback(
    [Output(f"collapse_{i}", "is_open") for i in range(1, 4)],
    [Input(f"button_collapse_{i}", "n_clicks") for i in range(1, 4)],
    [State(f"collapse_{i}", "is_open") for i in range(1, 4)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3, is_open1, is_open2,is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_1" and n1:
        return not is_open1, False, False
    elif button_id == "button_collapse_2" and n2:
        return False, not is_open2, False
    elif button_id == "button_collapse_3" and n3:
        return False, False, not is_open3
    return False, False, False


#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('ver_mapas_componentes_button','disabled'),
              Input('dropdown_atrib_names','value')
            )
def enable_ver_mapas_componentes_button(values):
    if ( values ):
        return False
    else:
        return True





#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('mapa_componentes_fig','figure'),
              Input('ver_mapas_componentes_button','n_clicks'),
              State('dropdown_atrib_names','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_fig(click,names):

    '''
    prueba para ver si dibuja esto
    z=[[1, 20, 30],
        [20, 1, 60],
        [30, 60, 1]]


    trace = dict(type='heatmap', z=z, colorscale = 'Jet')
    data=[trace]

    
    layout = {}
    
    layout['xaxis'] = {'range': [-0.5, 100]}
    layout['width'] = 700
    layout['height']= 700
    fig = dict(data=data, layout=layout)

    print('test_render_finalizadosooo')
    return fig
    '''



    som = Sesion.modelo
    with open('data_session.json') as json_file:
        session_data = json.load(json_file)

    tam_eje_x = session_data['som_tam_eje_x'] 
    tam_eje_y = session_data['som_tam_eje_y'] 
    nombres_columnas = session_data['columns_names']
    nombres_atributos = nombres_columnas[0:len(nombres_columnas)-1]
    lista_de_indices = []
    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    

    pesos = som.get_weights()


    rows = ceil(len(lista_de_indices)/4)
    # rows*4 matrix for visualization
    fig = make_subplots(rows=rows, 
                        cols=4, 
                        shared_xaxes=False, 
                        horizontal_spacing=0.1, 
                        vertical_spacing=0.05, 
                        subplot_titles=names, 
                        column_widths=None, 
                        row_heights=None)

    traces = []
    for i in lista_de_indices:
        #print('\n Atrib ',i, )
        #print(pesos[:,:,i].tolist())
        traces.append(go.Heatmap(z=pesos[:,:,i].tolist())
                    #zmax=zmax, 
                    #zmin=zmin, 
                    #coloraxis = 'coloraxis2')],
                    #rows=[i // math.ceil(math.sqrt(data.shape[1])) + 1 ],
                    #cols=[i % math.ceil(math.sqrt(data.shape[1])) + 1 ])
                    
                    )
    
    for i in range(1,rows+1):
        for j in range(1,5):
            if( (i+j-2) < len(lista_de_indices) ):
                fig.append_trace(traces[i+j-2],row=i,col=j)
            else:
                break
    

  
    #not working:
    #fig.add_traces(traces)

    '''
        [go.Heatmap(z=pesos[:,:,i].tolist())]
                    #zmax=zmax, 
                    #zmin=zmin, 
                    #coloraxis = 'coloraxis2')],
                    #rows=[i // math.ceil(math.sqrt(data.shape[1])) + 1 ],
                    #cols=[i % math.ceil(math.sqrt(data.shape[1])) + 1 ]
        )
    '''

     
    print('render finalizado')
    return fig
  