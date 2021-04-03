import plotly.colors as clrs
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly.graph_objs import graph_objs
import numpy as np

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
     





def make_annotations(data, colorscale, reversescale):
    """
    Get annotations for each cell of the heatmap with graph_objs.Annotation
    :rtype (list[dict]) annotations: list of annotations for each cell of
        the heatmap
    """
    #TODO: ELIMAR LAS LLLAMDA A LA FUNCION DE ABAJO
    #min_text_color, max_text_color = get_text_color( colorscale, reversescale)
    white = "#FFFFFF"
    black = "#000000"
    
    min_text_color = white
    max_text_color = black
    zmin = np.nanmin(data)
    zmax = np.nanmax(data)
    zmid = (zmax + zmin) / 2
    annotations = []
    
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
    return annotations






#Plot fig with titles and gsom size
def get_fig_div_with_info(fig,fig_id, title,tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None):
    '''

        neurona_padre: None or str tuple if it exits
    '''

    
    if(neurona_padre is not None):
        div_info_neurona_padre = html.Div(children = [
            dbc.Badge('Neurona padre:', pill=True, color="light", className="mr-1"),
            dbc.Badge(neurona_padre, pill=True, color="info", className="mr-1")
        ])
       
    else:
        div_info_neurona_padre= ''


    if(gsom_level is not None):
        div_info_nivel_gsom = html.Div(children = [
             dbc.Badge('Nivel '+ str(gsom_level), pill=True , color="info", className="mr-1")
        ])
    else:
        div_info_nivel_gsom = ''

    


    div_inf_grid = html.Div(children = [
        html.H3(title),

        html.Div(children= [
            div_info_nivel_gsom,
            div_info_neurona_padre
        ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-direction': 'column '}),

        html.Div(children= [
            dbc.Badge(tam_eje_horizontal, pill=True, color="info", className="mr-1"),
            dbc.Badge('x', pill=True, color="light", className="mr-1"),
            dbc.Badge(tam_eje_vertical, pill=True, color="info", className="mr-1"),
            dbc.Badge('neuronas.', pill=True, color="light", className="mr-1")
        ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'})
        
    ], style={'margin': '0 auto','width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center',
                'flex-wrap': 'wrap', 'flex-direction': 'column ' })


      
    children =[ div_inf_grid, dcc.Graph(id=fig_id,figure=fig)  ]
    '''
    div = html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )
    '''
    return children
