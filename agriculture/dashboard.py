import base64

import dash_bootstrap_components.themes
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd


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

    def graph_index(self, df_stats):
        app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        # logo = '/home/newmar/Downloads/python_projects/doutorado/agriculture/templates/logo.png'  # replace with your own image
        # vector = gpd.read_file('/home/newmar/Downloads/python_projects/doutorado/inputs/vector_tasca_test.gpkg')

        # fig2 = vector.plot()
        app.layout = dbc.Container(
            dbc.Row([

                dbc.Col([
                    html.Div([
                        html.H3("Unioeste geoprocessing laboratory - GEOLAB"),
                        html.H5("History indexes database for Agriculture")
                    ], style={}),
                    html.P("", style={"margin-top": "40px"}),
                    dbc.Row([
                        html.Div([
                            html.H6("Select Range of years"),
                            html.Br(),
                            dcc.RangeSlider(
                                min=0,
                                max=9,
                                marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
                                # value=5,
                            ),
                        ], style={}),
                        html.P("", style={"margin-top": "20px"}),
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Min value history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="media-indice-text"),
                                html.Span("Média Indice", className="card-text"),
                                html.H5(id="mean-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Mean value history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="max-indice-text"),
                                html.Span("Máxima Indice", className="card-text"),
                                html.H5(id="maxima-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),

                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Max value history Index", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="min-indice-text"),
                                html.Span("Minima Indice", className="card-text"),
                                html.H5(id="minima-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),

                    ]),

                    html.P("", style={"margin-top": "40px"}),

                    dcc.Graph(
                        id='graph_indexes',
                        figure=self.ndvi_index(df_stats))], md=5,
                    style={"padding": "25px", "background-color": "#242424"}),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(df_stats['index'].unique(), placeholder="Select index"), ]),

                        dbc.Col([
                            dcc.Dropdown(df_stats['index'].unique(), placeholder="Select index"), ])

                    ]

                    ),

                    dcc.Graph(
                        id='outro',
                        figure=self.all_indexes(df_stats),
                        style={"height": "100vh"})
                ], md=7)], class_name='g-O'), fluid=True)

        app.run_server(debug=True)
