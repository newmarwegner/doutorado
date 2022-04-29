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

    def graph_index(self, df_stats):
        app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        logo = '/home/newmar/Downloads/python_projects/doutorado/agriculture/templates/logo.png'  # replace with your own image
        vector = gpd.read_file('/home/newmar/Downloads/python_projects/doutorado/inputs/vector_tasca_test.gpkg')
        fig = px.line(df_stats, x="data", y="mean", color="index", title='History Indexes in Study Area',
                      template='plotly_dark', markers=True)
        fig.update_layout(
            # mapbox_accesstoken=token,
            paper_bgcolor="#242424",
            autosize=True,
            margin=go.layout.Margin(l=10, r=10, t=30, b=10),
            showlegend=True, )
        # fig2 = vector.plot()
        app.layout = dbc.Container(
            dbc.Row([

                dbc.Col([
                    html.Div([
                        html.H3("Laboratório de Geoprocessamento da Unioeste - GEOLAB"),
                        html.H5("Base de dados de índice históricos para agricultura")
                    ], style={}),
                    html.P("", style={"margin-top": "40px"}),
                    dbc.Row([
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Média do Indice", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="media-indice-text"),
                                html.Span("Média Indice", className="card-text"),
                                html.H5(id="mean-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),
                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Máxima do Indice", className="card-text"),
                                html.H3(style={"color": "#adfc92"}, id="max-indice-text"),
                                html.Span("Máxima Indice", className="card-text"),
                                html.H5(id="maxima-indice-text"),
                            ])
                        ], color="light", outline=True, style={"margin-top": "10px",
                                                               "box-shadow": "0 4px 4px 0 rgba(0, 0, 0, 0.15), 0 4px 20px 0 rgba(0, 0, 0, 0.19)",
                                                               "color": "#FFFFFF"})], md=4),

                        dbc.Col([dbc.Card([
                            dbc.CardBody([
                                html.Span("Minima do Indice", className="card-text"),
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
                        figure=fig)], md=5, style={"padding": "25px", "background-color": "#242424"}),

                dbc.Col([
                    dcc.Graph(
                        id='outro',
                        figure=fig)
                ], md=7)], class_name='g-O'), fluid=True)

        app.run_server(debug=True)
