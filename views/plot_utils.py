import plotly.colors as clrs
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly.graph_objs import graph_objs
import numpy as np
from  config.config import DEFAULT_HEATMAP_COLORSCALE, DEFAULT_HEATMAP_PX_HEIGHT, DEFAULT_HEATMAP_PX_WIDTH, DISCRETE_COLORSCALE_269, DISCRETE_COLORSCALE_64


def get_DISCRETE_COLORSCALE_269():
    return  ['#000000','#FFFF00','#1CE6FF','#FF34FF','#FF4A46','#008941','#006FA6','#A30059','#FFDBE5','#7A4900','#0000A6','#63FFAC','#B79762','#004D43','#8FB0FF','#997D87','#5A0007','#809693','#FEFFE6','#1B4400','#4FC601','#3B5DFF','#4A3B53','#FF2F80','#61615A','#BA0900','#6B7900','#00C2A0','#FFAA92','#FF90C9','#B903AA','#D16100','#DDEFFF','#000035','#7B4F4B','#A1C299','#300018','#0AA6D8','#013349','#00846F','#372101','#FFB500','#C2FFED','#A079BF','#CC0744','#C0B9B2','#C2FF99','#001E09','#00489C','#6F0062','#0CBD66','#EEC3FF','#456D75','#B77B68','#7A87A1','#788D66','#885578','#FAD09F','#FF8A9A','#D157A0','#BEC459','#456648','#0086ED','#886F4C','#34362D','#B4A8BD','#00A6AA','#452C2C','#636375','#A3C8C9','#FF913F','#938A81','#575329','#00FECF','#B05B6F','#8CD0FF','#3B9700','#04F757','#C8A1A1','#1E6E00','#7900D7','#A77500','#6367A9','#A05837','#6B002C','#772600','#D790FF','#9B9700','#549E79','#FFF69F','#201625','#72418F','#BC23FF','#99ADC0','#3A2465','#922329','#5B4534','#FDE8DC','#404E55','#0089A3','#CB7E98','#A4E804','#324E72','#6A3A4C','#83AB58','#001C1E','#D1F7CE','#004B28','#C8D0F6','#A3A489','#806C66','#222800','#BF5650','#E83000','#66796D','#DA007C','#FF1A59','#8ADBB4','#1E0200','#5B4E51','#C895C5','#320033','#FF6832','#66E1D3','#CFCDAC','#D0AC94','#7ED379','#012C58','#7A7BFF','#D68E01','#353339','#78AFA1','#FEB2C6','#75797C','#837393','#943A4D','#B5F4FF','#D2DCD5','#9556BD','#6A714A','#001325','#02525F','#0AA3F7','#E98176','#DBD5DD','#5EBCD1','#3D4F44','#7E6405','#02684E','#962B75','#8D8546','#9695C5','#E773CE','#D86A78','#3E89BE','#CA834E','#518A87','#5B113C','#55813B','#E704C4','#00005F','#A97399','#4B8160','#59738A','#FF5DA7','#F7C9BF','#643127','#513A01','#6B94AA','#51A058','#A45B02','#1D1702','#E20027','#E7AB63','#4C6001','#9C6966','#64547B','#97979E','#006A66','#391406','#F4D749','#0045D2','#006C31','#DDB6D0','#7C6571','#9FB2A4','#00D891','#15A08A','#BC65E9','#FFFFFE','#C6DC99','#203B3C','#671190','#6B3A64','#F5E1FF','#FFA0F2','#CCAA35','#374527','#8BB400','#797868','#C6005A','#3B000A','#C86240','#29607C','#402334','#7D5A44','#CCB87C','#B88183','#AA5199','#B5D6C3','#A38469','#9F94F0','#A74571','#B894A6','#71BB8C','#00B433','#789EC9','#6D80BA','#953F00','#5EFF03','#E4FFFC','#1BE177','#BCB1E5','#76912F','#003109','#0060CD','#D20096','#895563','#29201D','#5B3213','#A76F42','#89412E','#1A3A2A','#494B5A','#A88C85','#F4ABAA','#A3F3AB','#00C6C8','#EA8B66','#958A9F','#BDC9D2','#9FA064','#BE4700','#658188','#83A485','#453C23','#47675D','#3A3F00','#061203','#DFFB71','#868E7E','#98D058','#6C8F7D','#D7BFC2','#3C3E6E','#D83D66','#2F5D9B','#6C5E46','#D25B88','#5B656C','#00B57F','#545C46','#866097','#365D25','#252F99','#00CCFF','#674E60','#FC009C','#92896B']

def get_DISCRETE_COLORSCALE_64():

    return  ['#000000', '#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
                            '#FFDBE5', '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
                            '#5A0007', '#809693', '#FEFFE6', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80',
                            '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100',
                            '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
                            '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09',
                            '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66',
                            '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C',
                            '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81',
                            '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
                            '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700',
                            '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329',
                            '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C']




def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    #bvals = sorted(bvals)     
    #nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    nvals = bvals
    dcolorscale = [] #discrete colorscale
    ticks_val = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
        #ticks_val.append( round((nvals[k] + nvals[k+1])/2,1) )
        ticks_val.append( (nvals[k] + nvals[k+1])/2 )

    return dcolorscale ,ticks_val  





#TODO MIRAR SI UTILIZO FIANLMENTE ESTAS TRES FUNCIONES
def to_rgb_color_list(color_str, default):
    if "rgb" in color_str:
        return [int(v) for v in color_str.strip("rgb()").split(",")]
    elif "#" in color_str:
        return clrs.hex_to_rgb(color_str)
    else:
        return default

def should_use_black_text(background_color):
  return (
      background_color[0] * 0.299
      + background_color[1] * 0.587
      + background_color[2] * 0.114
  ) > 186
  
def get_text_color(colorscale, reversescale):
      """
      Get font color for annotations.
      The annotated heatmap can feature two text colors: min_text_color and
      max_text_color. The min_text_color is applied to annotations for
      heatmap values < (max_value - min_value)/2. The user can define these
      two colors. Otherwise the colors are defined logically as black or
      white depending on the heatmap's colorscale.
      :rtype (string, string) min_text_color, max_text_color: text
          color for annotations for heatmap values <
          (max_value - min_value)/2 and text color for annotations for
          heatmap values >= (max_value - min_value)/2
      """
      # Plotly colorscales ranging from a lighter shade to a darker shade
      colorscales = [
          "Greys",
          "Greens",
          "Blues",
          "YIGnBu",
          "YIOrRd",
          "RdBu",
          "Picnic",
          "Jet",
          "Hot",
          "Blackbody",
          "Earth",
          "Electric",
          "Viridis",
          "Cividis",
      ]
      # Plotly colorscales ranging from a darker shade to a lighter shade
      colorscales_reverse = ["Reds"]
      white = "#FFFFFF"
      black = "#000000"
    
      min_text_color = white
      max_text_color = black
     
      if isinstance(colorscale, list):
          min_col = to_rgb_color_list(colorscale[0][1], [255, 255, 255])
          max_col = to_rgb_color_list(colorscale[-1][1], [255, 255, 255])
          # swap min/max colors if reverse scale
          if reversescale:
              min_col, max_col = max_col, min_col
          if should_use_black_text(min_col):
              min_text_color = black
          else:
              min_text_color = white
          if should_use_black_text(max_col):
              max_text_color = black
          else:
              max_text_color = white
      else:
          min_text_color = black
          max_text_color = black
      return min_text_color, max_text_color
     





def make_annotations(data, colorscale, reversescale= False, text=None):
    """
    Get annotations for each cell of the heatmap with graph_objs.Annotation
    :rtype (list[dict]) annotations: list of annotations for each cell of
        the heatmap
    """
    #TODO: ELIMAR LAS LLLAMDA A LA FUNCION DE ABAJO
    #min_text_color, max_text_color = get_text_color( colorscale, reversescale)
    white = "#FFFFFF"
    black = "#000000"
    
    if reversescale:
        min_text_color =  black
        max_text_color =  white   
    else:
        min_text_color = white
        max_text_color = black

    zmin = np.nanmin(data)
    zmax = np.nanmax(data)
    zmid = (zmax + zmin) / 2
    annotations = []

    if(text is None):#numeric data
        #print('texto',self.annotation_text[0][0])
        for n, row in enumerate(data):
            for m, val in enumerate(row):
                if(not np.isnan(val) ):
                    font_color = min_text_color if ( val < zmid ) else max_text_color
                    annotations.append(
                        graph_objs.layout.Annotation(
                               #text= '' if (np.isnan(val) ) else str(val) ,
                               text= str(val) ,
                               x=m,
                               y=n,
                               #xref="x1",
                               #yref="y1",
                            font=dict(color=font_color),
                            showarrow=False,
                        )
                    )


    else:#categorical data
      
        for n, row in enumerate(data):
            for m, val in enumerate(row):
                if(not np.isnan(val) ):
                    font_color = min_text_color if ( val < zmid ) else max_text_color
                    annotations.append(
                        graph_objs.layout.Annotation(
                               #text= '' if (np.isnan(val) ) else str(val) ,
                               text= text[n][m] ,
                               x=m,
                               y=n,
                               #xref="x1",
                               #yref="y1",
                            font=dict(color=font_color),
                            showarrow=False,
                        )
                    )
        

    return annotations

def fig_add_annotations(figure):
    data = figure['data']
    trace = data[0]
    #print(trace)
    data_to_plot = trace['z'] 
    #To replace None values with NaN values
    data_to_plot_1 = np.array(data_to_plot, dtype=np.float64)
    annotations = make_annotations(data_to_plot_1, colorscale = DEFAULT_HEATMAP_COLORSCALE, reversescale= False)
    layout = figure['layout']
    layout['annotations'] = annotations
    fig_updated = dict(data=data, layout=layout)
    return fig_updated

def fig_del_annotations(figure):
    data = figure['data']
    layout = figure['layout']
    layout['annotations'] = []
    fig_updated = dict(data=data, layout=layout)
    return fig_updated



#Plot fig with titles and gsom size
def get_fig_div_with_info(fig,fig_id, title,tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None):
    '''

        neurona_padre: None or str tuple if it exits
    '''
    if(neurona_padre is not None):
        div_info_neurona_padre = html.Div(children = [
            dbc.Badge('Parent Neuron:', pill=True, color="light", className="mr-1"),
            dbc.Badge(neurona_padre, pill=True, color="info", className="mr-1")
        ])
    else:
        div_info_neurona_padre= ''

    if(gsom_level is not None):
        div_info_nivel_gsom = html.Div(children = [
             dbc.Badge('Level '+ str(gsom_level), pill=True , color="info", className="mr-1")
        ])
    else:
        div_info_nivel_gsom = ''


    if(tam_eje_horizontal is None or tam_eje_vertical  is None ):
        div_info_dimensions = ''
    else:
        div_info_dimensions =  html.Div(children= [
                dbc.Badge(tam_eje_horizontal, pill=True, color="info", className="mr-1"),
                dbc.Badge('x', pill=True, color="light", className="mr-1"),
                dbc.Badge(tam_eje_vertical, pill=True, color="info", className="mr-1"),
                dbc.Badge('neurons.', pill=True, color="light", className="mr-1")
            ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'})



    div_inf_grid = html.Div(children = [
        html.H3(title),

        html.Div(children= [
            div_info_nivel_gsom,
            div_info_neurona_padre
        ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-direction': 'column '}),
        div_info_dimensions
       
        
    ], style={'margin': '0 auto','width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center',
                'flex-wrap': 'wrap', 'flex-direction': 'column ' })


    children =[ div_inf_grid, dcc.Graph(id=fig_id,figure=fig)  ]
    '''
    div = html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )
    '''
    return children





def create_heatmap_figure(data,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = None, 
                            colorscale =DEFAULT_HEATMAP_COLORSCALE, reversescale=False, text = None,
                            discrete_values_range=None,unique_targets=None ):

    if(tam_eje_horizontal >tam_eje_vertical ):
        xaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
        yaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
    else:
        yaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_vertical-1)+0.5] , 'constrain' : "domain"}
        xaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'y','scaleratio': 1 }

    layout = { 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict}
    layout['width'] = DEFAULT_HEATMAP_PX_WIDTH
    layout['height']= DEFAULT_HEATMAP_PX_HEIGHT
     
    #condition_Nones = not(val is None)
    #condition_nans= not(np.isnan(val))

    if(title is not None):
        layout['title'] = title

 
    if(text is None):
        if(check_annotations):
            annotations = make_annotations(data, colorscale = colorscale, reversescale= reversescale)
            layout['annotations'] = annotations

        trace = dict(type='heatmap', z=data, colorscale = colorscale,reversescale= reversescale)
    else:

        if(check_annotations):
            annotations = make_annotations(data, colorscale = colorscale, reversescale= reversescale, text= text)
            layout['annotations'] = annotations

        if(len(discrete_values_range) >=270  ):
            #no colorbar,no custom colorsacle too much categorial unique targets

            trace = dict(type='heatmap', z=data, 
                        showlegend=False,
                        showscale=False,
                        text=text,
                        hovertemplate= '%{text}</b><br>'
                            +'x: %{x} , y: %{y}<br>' 
                            +"<extra></extra>"
                )

        else:# we can set a color per each different target

            discrete_values_range.append( 1.0 )#cuadrar rangos en la funcion
            colores = get_DISCRETE_COLORSCALE_269()[0:( (len(discrete_values_range)-1) ) ]
            print('longitud de colres', len(colores))
            #print('discrete_values_range',discrete_values_range)
            #print('colores',colores)
            d_color_scale,tickvals = discrete_colorscale(bvals = discrete_values_range, colors = colores )

            #bvals = np.array(discrete_values_range[:-1])
            #tickvals = [np.mean(bvals[k:k+2]) for k in range(len(bvals)-1)]

            #print('escala:', d_color_scale)
            #print('unique targets',unique_targets)
            #print('lista corriente',['a','b','c'])
            #print('valores para ticks',tickvals)
            
            #print('data',data)
            #print('text',text)
            trace = dict(type='heatmap', z=data, 
                        #name = text,
                        zmin=0.0,
                        zmax=1.0,
                        #colorscale= [[0.0,'#0d0887'], [0.1,'#46039f'], [0.2,'#7201a8'], [0.3,'#9c179e'], [0.4,'#bd3786'], [0.5,'#d8576b'], [0.6,'#ed7953'], [0.7,'#fb9f3a'], [0.8,'#fdca26'], [1.0,'#f0f921']],
                        colorbar = dict(    thickness=25,
                                         tickvals=tickvals, 
                                         ticktext=unique_targets,             
                        ),
                        showlegend=False,

                        colorscale= d_color_scale,
                        text=text,
                        hovertemplate= '%{text}</b><br>'
                            +'x: %{x} , y: %{y}<br>' 
                            +"<extra></extra>"
            )



     


    data=[trace]
    data.append({'type': 'scattergl',
                    'mode': 'text'
                })

    figure = dict(data=data, layout=layout)


    

    return figure






    #TODO BORRAR ESTO SI LOS HEATMAPS VAN OK
    '''
 

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




def get_single_heatmap_css_style():
    style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                        'align-items': 'center', 'justify-content': 'center',
                                        'flex-wrap': 'wrap', 'flex-direction': 'column ' }
    return style

