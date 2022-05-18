import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import islice
from dash import Dash, dcc, html, Output, Input
import warnings
warnings.filterwarnings("ignore")

class Dashboard:
    def __init__(self):
        pass

    def ndvi_index(self, df_stats):
        fig = px.line(df_stats[df_stats['index'] == 'ndvi'], x="data", y="mean",
                      title='History of NDVI Index in Study Area',
                      template='plotly_dark', markers=False)

        fig.update_layout(
            paper_bgcolor="#242424",
            autosize=True,
            margin=go.layout.Margin(l=100, r=100, t=100, b=0),
            showlegend=True, )
        return fig

    def all_indexes(self, df_stats):
        fig = px.line(df_stats, x="data", y="mean",
                      # title='History of NDVI Index in Study Area',
                      template='plotly_dark', color='index', markers=False)

        fig.update_layout(
            paper_bgcolor="#242424",
            autosize=True,
            margin=go.layout.Margin(l=100, r=100, t=20, b=80),
            showlegend=True, )
        return fig

    def heatmap_db(self, df_rasters, index='ndvi', data=pd.Timestamp('2018-12-17')):
        df = df_rasters.loc[df_rasters['index'] == index]
        df['data'] = pd.to_datetime(df['data'])
        df = df.loc[df['data'] == data]
        width = pd.DataFrame(df['raster_profile']).to_dict('records')[0]['raster_profile']['width']
        input = list(df['raster_array'])[0]
        output = list(islice(input, width))
        fig = px.imshow(output, template='plotly_dark')

        fig.update_layout(
            paper_bgcolor="#242424",
            autosize=True,
            margin=go.layout.Margin(l=100, r=100, t=20, b=80),
            showlegend=True, )
        return fig

    def graph_index(self, df_stats, df_rasters):
        # logo = '/home/newmar/Downloads/python_projects/doutorado/agriculture/templates/logo.png'  # replace with your own image
        # vector = gpd.read_file('/home/newmar/Downloads/python_projects/doutorado/inputs/vector_tasca_test.gpkg')
        app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

        app.layout = dbc.Container(
            dbc.Row([

                dbc.Col([
                    html.Div([
                        html.H3("Geoprocessing laboratory - GEOLAB"),
                        html.H5("History indexes database for Agriculture")
                    ], style={}),
                    html.P("", style={"margin-top": "40px"}),
                    dbc.Row([
                        html.Div([
                            html.H6("Select Range of years"),
                            html.Br(),
                            dcc.RangeSlider(id='input_range',
                                            min=pd.DatetimeIndex(df_stats['data']).year.unique().min(),
                                            max=pd.DatetimeIndex(df_stats['data']).year.unique().max(),
                                            marks={i: str(i) for i in pd.DatetimeIndex(df_stats['data']).year.unique()},

                                            ),
                        ], style={}),
                        html.P("", style={"margin-top": "20px"}),
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Min value NDVI history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="min-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Mean value NDVI history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="mean-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),

                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Max value NDVI history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="max-indice-text"),

                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),

                    ]),

                    html.P("", style={"margin-top": "40px"}),

                    dcc.Graph(
                        id='graph_indexes',
                    )], md=5,
                    style={"padding": "25px", "background-color": "#242424"}),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(df_stats['index'].unique(), 'ndvi', placeholder="Select index",
                                         id='dropdown_a')]),

                        dbc.Col([
                            dcc.Dropdown(df_stats['index'].unique(), 'avi', placeholder="Select index",
                                         id='dropdown_b'), ])

                    ]),

                    dcc.Graph(
                        id='all_indexes',
                        style={"height": "45vh"}),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(df_stats['index'].unique(), 'ndvi', placeholder="Select index",
                                         id='dropdown_index')]),
                        dbc.Col([
                            dcc.Dropdown(df_stats['data'].unique(),df_stats['data'].unique()[0], placeholder="Select date",
                                         id='dropdown_date')]),
                    ]),
                    dcc.Graph(
                        id='raster_map',
                        style={"height": "45vh"})
                ], md=7)], class_name='g-O'), fluid=True)

        @app.callback(
            [Output(component_id='graph_indexes', component_property='figure'),
             Output(component_id='min-indice-text', component_property='children'),
             Output(component_id='mean-indice-text', component_property='children'),
             Output(component_id='max-indice-text', component_property='children')],
            Input(component_id='input_range', component_property='value')
        )
        def update_ndvi_index(input_value):
            if input_value is None:
                df = df_stats
            else:
                dateindex = (pd.DatetimeIndex(df_stats['data']).year >= input_value[0]) & (
                        pd.DatetimeIndex(df_stats['data']).year <= input_value[1])
                df = df_stats.loc[dateindex]
            filter = df['index'] == 'ndvi'
            ndvi_mean = df.loc[filter]['mean'].mean()
            ndvi_max = df.loc[filter]['mean'].max()
            ndvi_min = df.loc[filter]['mean'].min()

            return [self.ndvi_index(df), str(round(ndvi_min, 3)), str(round(ndvi_mean, 3)), str(round(ndvi_max, 3))]

        @app.callback(
            Output(component_id='all_indexes', component_property='figure'),
            [Input(component_id='input_range', component_property='value'),
             Input(component_id='dropdown_a', component_property='value'),
             Input(component_id='dropdown_b', component_property='value'),
             ]
        )
        def update_all_index(input_range, dropdown_a, dropdown_b):
            indexes = [dropdown_a, dropdown_b]

            if input_range is None:
                df = df_stats.loc[df_stats['index'].isin(indexes)]
            else:
                dateindex = (pd.DatetimeIndex(df_stats['data']).year >= input_range[0]) & (
                        pd.DatetimeIndex(df_stats['data']).year <= input_range[1]) & df_stats['index'].isin(indexes)
                df = df_stats.loc[dateindex]

            return self.all_indexes(df)

        @app.callback(
            Output(component_id='raster_map', component_property='figure'),
            [Input(component_id='dropdown_index', component_property='value'),
             Input(component_id='dropdown_date', component_property='value'),]
        )
        def update_heatmap_db(dropdown_index, dropdown_date):
            dropdown_date = pd.Timestamp(dropdown_date)

            return self.heatmap_db(df_rasters,dropdown_index,dropdown_date)

        app.run_server(debug=True)
