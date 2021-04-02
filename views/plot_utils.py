import plotly.colors as clrs
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