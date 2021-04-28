# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements


import io
from io import BytesIO
from datetime import datetime
import base64

import pandas as pd
import numpy as np

from  views.session_data import session_data
from  config.config import *
from  config.config import DEFAULT_DATASET_ROWS_PREVIEW


import plotly.graph_objects as go






#############################################################
#	                       LAYOUT	                        #
#############################################################

def Home(): 

    session_data.clean_session_data()
    layout = html.Div(children=[

        html.Div(id="hidden_div_for_redirect_callback"),

        #Dash componets for storing data
        dcc.Store(id='original_dataframe_storage',data=None),
        dcc.Store(id='processed_dataframe_storage',data=None),
        dcc.Store(id='notnumeric_dataframe_storage',data=None),
        dcc.Store(id='trainready_dataframe_storage',data=None),


        elements.navigation_bar,
        elements.cabecera,

        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Select Dataset')),

            dbc.CardBody(
                dbc.ListGroup([
                    # Archivo Local
                    dbc.ListGroupItem([
                        html.H4('Local File',className="card-title" , style={'textAlign': 'center'} ),
                        dcc.Upload( id='upload-data', children=html.Div(['Drag and Drop or  ', html.A('Select File  (.csv or .xls)')]),
                                            style={'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'},
                                            # Allow multiple files to be uploaded
                                            multiple=False),
                        #info showed when the dataset its loaded
                        html.Div(id='info_dataset_div',
                                style={'textAlign': 'center', "visibility": "hidden",'display':'none'} ,
                                children = div_info_dataset('','', '', '', None ) 
                        ), 

                        html.Div(id='hidden_div',children= '' ), 

                        html.Div(id='hidden_div_forcontinue',children = ''),
                        html.Div( 
                            [dbc.Button("Analyze Data", id="continue_button_home",disabled= True,
                            href=URLS['TRAINING_SELECTION_URL'], className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),

                 
                    
                ],flush=True,),


            )
        ]),

    



    ])


    return layout








#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################

 
def div_info_dataset(filename,fecha_modificacion, n_samples, n_features, df):
    if(df is None):
        return ''
    return html.Div(id='output_uploaded_file',children=[
                html.P(children= 'File:  ' + filename, style={'textAlign': 'center'} ),
                html.P(children= 'Last modified: ' + fecha_modificacion, style={'textAlign': 'center'} ),
                html.P(children= 'Number of Samples:  ' + n_samples , style={'textAlign': 'center'} ),
                html.P(children= 'Number of Features:  ' + n_features , style={'textAlign': 'center'} ),


                html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
                html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
                html.Div(id='n_samples',style={'textAlign': 'center'} ),
                html.Div(id='n_features',style={'textAlign': 'center'} ),
                dbc.RadioItems(
                    options=[
                        {"label": "Discret Target", "value": 1},
                        {"label": "Continuous Target", "value": 2},
                    ],
                    value=1,
                    id="radio_discrete_continuous",
                ),


                html.Hr(),


                #Atrib names
                html.H6('Feature Selection:'
                       
                ),

                dcc.Dropdown(id='dropdown_col_df_names',
                           options=[],
                           multi=True,
                           value = []
                ),


                dbc.Checklist(
                    options=[
                        {"label": "Detect and Clean Categorical Data", "value": 1},
                    ],
                    value=[],
                    id="clean_data_switch",
                    switch=True,
                ),

                html.Hr(),
                dbc.Checklist(  options=[{"label": "Apply One Hot Encoding", "value": 0}],
                                            value=[],
                                            id="check_onehot"),


                html.Div(id='div_onehot_menu',children=get_onehot_childrendiv_menu()),

                html.Hr(),

                html.H4('Table Preview',className="card-title" , style={'textAlign': 'center'} ),

            
                
                #html.Div(id = 'preview_table' ,children =create_preview_table(df))
                html.Div(id = 'preview_table' ,children ='')

                

            ])
                      



def create_preview_table(df):

    selected_columns = []
    if (len(df.columns) > 0):
        selected_columns.append(df.columns[len(df.columns)-1])
   
    return html.Div([

                    dcc.Loading(id='loading',
                                type='circle',
                                children=[
                                 

                                    html.Div(style = {"overflow": "scroll"},
                                        children=
                                        dash_table.DataTable(
                                            id='dataset_table_preview',
                                            column_selectable="single",
                                            columns=[{"name": i, "id": i, "selectable": True} for i in df.columns],
                                            
                                            selected_columns=selected_columns,
                                            data=df.head(DEFAULT_DATASET_ROWS_PREVIEW).to_dict('records'),

                                            style_cell={'textAlign': 'center',
                                                        'textOverflow': 'ellipsis',
                                                        'overflowX': 'auto'
                                            },



                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgb(248, 248, 248)'
                                                }
                                            ],

                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold'
                                            }
                                        )

                                    )


                                ]
                                    
                                
                    )

    ])


def get_onehot_childrendiv_menu():

    collapse = dbc.Collapse(id="collapse_onehot_menu",children=
        dbc.CardBody(children=[ 
            html.Div(

                children = [
                
                
                    html.H5("Select features to apply One Hot:"),
                    dcc.Dropdown(
                        id='dropdown_atrib_names_home',
                        options=[],
                        multi=True
                    ),

                    dbc.Checklist(  options=[{"label": "Consider empty values", "value": 0}],
                                                        value=[],
                                                        id="check_nanvalues_onehot"
                    ),

                    dbc.Button("Apply One Hot", id="apply_onehot_button", className="mr-2", color="primary")


                ]
            )
        ])
    )

    return collapse




def getNumericVars(df, clean_categorical_data):
    #Only numeric variables
    df_numeric = df.select_dtypes(['number'])

    if(clean_categorical_data):
        # Auto detect categorical variables to exclude them
        allvars = set(df_numeric.columns)
        intvars = df_numeric.columns[df_numeric.dtypes == 'int']
        categorical = []
        for col in intvars:
            if(len(df_numeric[col].unique())<15):
                categorical.append(col)

        catvars = set(categorical)
        numericvars = allvars-catvars
        df_numeric = df_numeric[numericvars]
        df_categorial = df_numeric[catvars]
    else:
        df_categorial = pd.DataFrame(columns= [])


    df_not_numeric = df.select_dtypes(exclude = 'number')
    #print(df_not_numeric)
    frames = [df_not_numeric, df_categorial ]
    df_not_numeric = pd.concat(frames, axis=1)

    return df_numeric, df_not_numeric



#############################################################
#	                     CALLBACKS	                        #
#############################################################


#Seleccionar columna
@app.callback(
    Output('dataset_table_preview', 'style_data_conditional'),
    Input('dataset_table_preview', 'selected_columns')
)
def update_styles(selected_columns):
    #print('selected coluss', selected_columns)
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]



@app.callback(  Output('dropdown_atrib_names_home', 'options'),
                Input('notnumeric_dataframe_storage', 'data'),
                prevent_initial_call=True 
)
def onehot_dropdown_options(input_data):

    options = []  # must be a list of dicts per option
    if input_data is not None:
        dff = pd.read_json(input_data,orient='split')
        col = dff.columns
         
        for n in col:
            options.append({'label' : n, 'value': n})
    return options




#Process data
@app.callback(  Output('processed_dataframe_storage','data'),
                Output('notnumeric_dataframe_storage','data'),
                Input('original_dataframe_storage','data'),
                Input('clean_data_switch', 'value'),    
                Input('apply_onehot_button', 'n_clicks'),
                State('processed_dataframe_storage','data'),
                State('notnumeric_dataframe_storage','data'),
                State('dropdown_atrib_names_home','value'),
                State('check_nanvalues_onehot','value'),
                prevent_initial_call=True
            )
#TODO MEMORIZE
def process_data(input_data , clean_categorical_data, n_clicks, processed_data, notnum_data, names,check_nan):


    if input_data is not None:

        ctx = dash.callback_context
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        #contexto de apply onehot: modificamos processed y notnumdata
        if(button_id == 'apply_onehot_button'): 

            notnum_df   = pd.read_json(notnum_data,orient='split')

            
            if( (processed_data is None) or (not names) or (notnum_df.empty ) ):
               #devolvemos tal cual pq no hay cambios
               return  processed_data,notnum_data


            else:#applicamos onehot

                processed_df = pd.read_json(processed_data,orient='split')

                for n in names:
                    # use pd.pd.concat to join the new columns with your original dataframe
                    processed_df = pd.concat( [pd.get_dummies(notnum_df[n], prefix=str(n), dummy_na=check_nan), processed_df ],axis=1)
                    notnum_df.drop([n],axis=1, inplace=True)


                processed_data = processed_df.to_json(date_format='iso',orient = 'split')
                notnum_data = notnum_df.to_json(date_format='iso',orient = 'split')
                return  processed_data,notnum_data
       

       #contexto del switch o del original_Dataframe: procesamos datos
        else:  

            dff = pd.read_json(input_data,orient='split')
            dff_num, dff_not_num = getNumericVars(dff,bool(clean_categorical_data))

            processed_data = dff_num.to_json(date_format='iso',orient = 'split')
            notnum_data = dff_not_num.to_json(date_format='iso',orient = 'split')

            return  processed_data,notnum_data

    
    else:
        dff = pd.DataFrame(columns=[])
        return dff,dff










#Show One Hot Encoding Menu
@app.callback(Output('collapse_onehot_menu', 'is_open'),
              Input('check_onehot', 'value'),
              prevent_initial_call=True )
def show_onehot_menu(check_onehot):
    if(check_onehot):
        return True
    else:
        return False


#Discrete/cont. data
@app.callback(Output('hidden_div', 'children'),
              Input('radio_discrete_continuous', 'value'),
              prevent_initial_call=True)
def update_target_type(radio_option):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (trigger_id == "radio_discrete_continuous"):
        
        if(radio_option == 1):
            session_data.set_discrete_data(True)
        else:
            session_data.set_discrete_data(False)

        return  ''



#Carga la info del dataset en el home
@app.callback(Output('original_dataframe_storage', 'data'),
              Output('info_dataset_div', 'children'),
              Output('info_dataset_div', 'style'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), 
              prevent_initial_call=True)
#TODO
#@cache.memoize(timeout=60)  # in seconds
def update_output( contents, filename, last_modified):
    '''Carga el dataset en los elementos adecuados

    '''
    if contents is not None:
        
        show_file_info_style =  {'textAlign': 'center',  'display': 'block'}

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                try:
                    # Assume that the user uploaded a CSV file
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = None, decimal = ",", engine='python')

                except: 
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = ',', decimal = ".")

            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                dataframe = pd.read_excel(io.BytesIO(decoded))

            else:
                return None,html.Div([ 'ERROR: File format not admited']),show_file_info_style

        except Exception as e:
            print(e)
            return None,html.Div([ 'An error occurred processing the file']), show_file_info_style
        
    

        n_samples, n_features=dataframe.shape
        if(n_samples == 0):
            return None, html.Div([ 'ERROR: The file does not contain any  sample']),show_file_info_style
        elif(n_features<=1):
            return None,html.Div([ 'ERROR: The file does not contain any feature']),show_file_info_style

      
        divv = div_info_dataset(filename,
                                datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S'),
                                str(n_samples),
                                str(n_features),
                                dataframe) 
        return dataframe.to_json(date_format='iso',orient = 'split'),divv, show_file_info_style
                

    else: 
        return  None,div_info_dataset('','', '', '' , None)  ,{'textAlign': 'center', "visibility": "hidden"}






# Select columns and store in filtered-data-storage
@app.callback(  Output('preview_table','children'),
                Output('trainready_dataframe_storage','data'),
                Output('continue_button_home','disabled'),
                Input('processed_dataframe_storage','data'),
                Input('dropdown_col_df_names','value'),
                prevent_initial_call=True
               )
def update_table_preview(input_data,columns):

    if input_data is not None and len(columns)>0:
        dff = pd.read_json(input_data,orient='split')
        dff = dff[columns]
        disabled_button = False

    else:
        dff =   pd.DataFrame(columns=[])
        disabled_button = True


    data =  dff.to_json(date_format='iso',orient = 'split')
    table =  create_preview_table(dff)


    return table,data,disabled_button


# numerical_features_to_dropdown
@app.callback(  Output('dropdown_col_df_names','options'),
                Output('dropdown_col_df_names','value'),
                Input('processed_dataframe_storage','data'),
                prevent_initial_call=True
            )

def numerical_features_to_dropdown(input_data):

        if input_data is not None:
            dff = pd.read_json(input_data,orient='split')
            columns = dff.columns
        else:
            columns=[]


        options = []  # must be a list of dicts per option

        for n in columns:
            options.append({'label' : n, 'value': n})

        return options, columns
       



#Boton de continuar
@app.callback(Output('hidden_div_forcontinue', 'children'),
              Input('continue_button_home', 'n_clicks'),
              State('trainready_dataframe_storage','data'),
              State('dataset_table_preview', 'selected_columns'),
              prevent_initial_call=True)
def analizar_datos_home( n_clicks,data, selected_col ):

    selected_col_name = selected_col[0]
    session_data.set_target(selected_col_name)
    df = pd.read_json(data,orient='split')
    session_data.set_pd_dataframe(df)

    return ' '
