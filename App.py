import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY])

# --- BARRE DE NAVIGATION ---
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                # PARTIE IMAGE
                dbc.Col(html.Img(src="assets/img3.png", height="40px")),
                # PARTIE TITRE
                dbc.Col(dbc.NavbarBrand(" Microfinance Credit Risk Dashboard ", className="ms-2 fw-bold text-white")),
            ], align="center", className="g-0"),
            href="/",
            style={"textDecoration": "none"},
        ),
        dbc.Nav([
            dbc.NavLink("Exploration", href="/", active="exact", className="text-white"),
            dbc.NavLink("Modélisation", href="/modelisation", active="exact", className="text-white"),
        ], navbar=True),
    ], fluid=True),
    color="primary",
    dark=True,
    className="mb-4 shadow"
)

app.layout = html.Div([
    navbar,
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=False, port=8050, use_reloader=False)

