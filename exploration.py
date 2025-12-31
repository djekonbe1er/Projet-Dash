import dash
from dash import dcc, html, callback, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from utils.donne import load_data

# Enregistrement de la page
dash.register_page(__name__, path='/')

df = load_data()

# --- STYLE AVEC OMBRE BLEUE
style_card = {
    "boxShadow": "0 8px 20px rgba(0, 102, 204, 0.25)",
    "border": "1px solid rgba(0, 102, 204, 0.3)",
    "borderRadius": "15px",
    "backgroundColor": "white",
    "height": "100%",
    "padding": "15px"
}

labels_fr = {
    'dsti_pct': "Taux d'endettement (DSTI %)",
    'revenu_mensuel_xof': "Revenu Mensuel (FCFA)",
    'montant_pret_xof': "Montant du Prêt (FCFA)",
    'epargne_xof': "Épargne Client (FCFA)",
    'age': "Âge du Client",
    'jours_retard_12m': "Jours de retard (1 an)",
    'defaut_90j': "Statut de Défaut",
    'anciennete_relation_mois': "Ancienneté (Mois)",
    'region': "Région",
    'secteur_activite': "Secteur",
    'canal_octroi': "Canal"
}

num_cols = [c for c in df.select_dtypes(include=['number']).columns if c in labels_fr]

layout = dbc.Container([
    html.Div([
        html.H1("Tableau de Bord", className="fw-bold mt-4", style={"color": "#004085"}),
        html.P("Analyse exploratoire", className="text-muted mb-5")
    ], className="text-center"),

    # --- SECTION 1 : FILTRES ---
    dbc.Card([
        dbc.CardHeader("Filtres Globaux", className="bg-primary text-white fw-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Région", className="small fw-bold"),
                    dcc.Dropdown(id='f-region', options=[{'label': i, 'value': i} for i in df['region'].unique()],
                                 multi=True, className="shadow-sm")
                ], width=3),
                dbc.Col([
                    html.Label("Secteur", className="small fw-bold"),
                    dcc.Dropdown(id='f-secteur',
                                 options=[{'label': i, 'value': i} for i in df['secteur_activite'].unique()],
                                 multi=True, className="shadow-sm")
                ], width=3),
                dbc.Col([
                    html.Label("Canal", className="small fw-bold"),
                    dcc.Dropdown(id='f-canal', options=[{'label': i, 'value': i} for i in df['canal_octroi'].unique()],
                                 multi=True, className="shadow-sm")
                ], width=3),
                dbc.Col([
                    html.Label("Volume de Prêt", className="small fw-bold"),
                    dcc.RangeSlider(
                        id='f-montant',
                        min=df['montant_pret_xof'].min(),
                        max=df['montant_pret_xof'].max(),
                        value=[df['montant_pret_xof'].min(), df['montant_pret_xof'].max()],
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], width=3),
            ])
        ])
    ], style=style_card, className="mb-4 shadow"),

    # --- SECTION 2 : KPIs ---
    dbc.Row(id='kpi-container', className="mb-4"),

    # --- SECTION 3 : GRAPHIQUES ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribution de l'endettement", className="fw-bold text-primary"),
                dbc.CardBody(dcc.Graph(id='graph-dsti', style={"height": "350px"}))
            ], style=style_card)
        ], width=5),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Nuage des points", className="fw-bold text-primary"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            [dcc.Dropdown(id='x-axis', options=[{'label': labels_fr[i], 'value': i} for i in num_cols],
                                          value='revenu_mensuel_xof', className="mb-2 shadow-sm")]),
                        dbc.Col(
                            [dcc.Dropdown(id='y-axis', options=[{'label': labels_fr[i], 'value': i} for i in num_cols],
                                          value='montant_pret_xof', className="mb-2 shadow-sm")]),
                    ]),
                    dcc.Graph(id='graph-scatter', style={"height": "280px"})
                ])
            ], style=style_card)
        ], width=7),
    ], className="mb-4 d-flex align-items-stretch"),

    # Matrice de Corrélation
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Matrice de corrélation", className="fw-bold text-primary"),
            dbc.CardBody(dcc.Graph(id='graph-corr'))
        ], style=style_card), width=12)
    ], className="mb-4 shadow"),

    # --- SECTION 4 : TABLE DES DONNÉES ---
    html.H3("Détail de la base", className="text-secondary mt-5 mb-3"),
    dbc.Card([
        dbc.CardBody([
            dash_table.DataTable(
                id='table-exploration',
                columns=[{"name": labels_fr.get(i, i), "id": i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto', 'borderRadius': '10px'},
                style_cell={'textAlign': 'left', 'padding': '12px'},
                style_header={'backgroundColor': '#004085', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f2f7fb'},
                    {'if': {'filter_query': '{defaut_90j} eq 1'}, 'backgroundColor': '#fadbd8', 'color': '#922b21',
                     'fontWeight': 'bold'}
                ],
            )
        ])
    ], style=style_card, className="mb-5 shadow")

], fluid=True, style={"backgroundColor": "#f0f4f8", "minHeight": "100vh"})


@callback(
    [Output('kpi-container', 'children'),
     Output('graph-dsti', 'figure'),
     Output('graph-scatter', 'figure'),
     Output('graph-corr', 'figure'),
     Output('table-exploration', 'data')],
    [Input('f-region', 'value'),
     Input('f-secteur', 'value'),
     Input('f-canal', 'value'),
     Input('f-montant', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value')]
)
def update_dashboard(reg, sect, canal, montant, x_val, y_val):
    # 1. Filtrage sécurisé
    dff = df.copy()
    if reg: dff = dff[dff['region'].isin(reg)]
    if sect: dff = dff[dff['secteur_activite'].isin(sect)]
    if canal: dff = dff[dff['canal_octroi'].isin(canal)]
    if montant:
        dff = dff[(dff['montant_pret_xof'] >= montant[0]) & (dff['montant_pret_xof'] <= montant[1])]

    # 2. Sécurité si dff est vide
    if dff.empty:
        empty_fig = go.Figure().update_layout(title="Aucune donnée", template="plotly_white")
        return [dbc.Col(dbc.Alert("Ajustez vos filtres pour voir les données", color="info"))], empty_fig, empty_fig, empty_fig, []

    # 3. Calcul des KPIs
    tx_defaut = (dff['defaut_90j'].mean() * 100)
    nb_dossiers = len(dff)
    epargne_moy = dff['epargne_xof'].mean()

    kpis = [
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Taux de Défaut", className="small"), html.H3(f"{tx_defaut:.1f}%")])], color="#c0392b", inverse=True, style={"borderRadius": "15px", "boxShadow": "0 4px 12px rgba(192, 57, 43, 0.3)"})),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Dossiers Actifs", className="small"), html.H3(f"{nb_dossiers}")])], color="#2c3e50", inverse=True, style={"borderRadius": "15px", "boxShadow": "0 4px 12px rgba(44, 62, 80, 0.3)"})),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Épargne Moyenne", className="small"), html.H3(f"{epargne_moy:,.0f} FCFA")])], color="#27ae60", inverse=True, style={"borderRadius": "15px", "boxShadow": "0 4px 12px rgba(39, 174, 96, 0.3)"})),
    ]

    # --- CORRECTION COULEUR PROFESSIONNELLE ---
    fig_dsti = px.histogram(
        dff,
        x="dsti_pct",
        labels=labels_fr,
        template="plotly_white",
        color_discrete_sequence=['#004085'] # Un seul bleu marine profond
    )
    fig_dsti.update_layout(
        bargap=0.1, # Espace léger entre les barres pour la clarté
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Scatter
    fig_scatter = px.scatter(dff, x=x_val, y=y_val, color="defaut_90j", labels=labels_fr, color_continuous_scale="RdYlGn_r", template="plotly_white")
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)

    # Corrélation
    cols_corr = ['age', 'revenu_mensuel_xof', 'epargne_xof', 'jours_retard_12m', 'montant_pret_xof', 'dsti_pct', 'defaut_90j']
    df_corr_data = dff[cols_corr].corr()
    fig_corr = px.imshow(
        df_corr_data, text_auto=".2f",
        x=[labels_fr.get(c, c) for c in df_corr_data.columns],
        y=[labels_fr.get(c, c) for c in df_corr_data.index],
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto"
    )
    fig_corr.update_layout(height=500, margin=dict(l=150, r=50, t=50, b=100), xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)')

    return kpis, fig_dsti, fig_scatter, fig_corr, dff.to_dict('records')