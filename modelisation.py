import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_curve, auc
from utils.donne import load_data

dash.register_page(__name__, path='/modelisation')

# --- STYLE AVEC OMBRE BLEUE HARMONISÉ ---
style_card = {
    "boxShadow": "0 8px 20px rgba(0, 102, 204, 0.25)",  # Ombre bleue plus visible
    "border": "1px solid rgba(0, 102, 204, 0.3)",
    "borderRadius": "15px",
    "backgroundColor": "white",
    "height": "100%",  # Force la même hauteur pour les colonnes d'une même ligne
    "padding": "10px"
}

# --- 1. CHARGEMENT ET ENTRAÎNEMENT ---
df = load_data()
# On utilise les variables demandées
features = ['age', 'revenu_mensuel_xof', 'epargne_xof', 'jours_retard_12m', 'dsti_pct', 'montant_pret_xof']
X = df[features]
y = df['defaut_90j']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis().fit(X_train_s, y_train)
qda = QuadraticDiscriminantAnalysis().fit(X_train_s, y_train)


# Métriques réelles sur Test
def calc_m(model):
    preds = model.predict(X_test_s)
    return {
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "rec": recall_score(y_test, preds)
    }


m_lda = calc_m(lda)
m_qda = calc_m(qda)


# --- 2. FIGURES ---
def create_roc_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color="#bdc3c7"), showlegend=False))

    for model, name, color in zip([lda, qda], ["FDA (Linéaire)", "QDA (Courbe)"], ["#007bff", "#34495e"]):
        probs = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc(fpr, tpr):.2f})", mode='lines',
                                 line=dict(color=color, width=3)))

    fig.update_layout(title="Analyse de Performance ROC", template="plotly_white", margin=dict(t=40, b=40, l=40, r=40),
                      height=350)
    return fig


def create_cm_fig(model, title, color_scale):
    preds = model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds)
    fig = px.imshow(cm, text_auto=True, x=['Sain', 'Défaut'], y=['Sain', 'Défaut'],
                    color_continuous_scale=color_scale, title=title)
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=40, l=40, r=40), height=300)
    return fig


# --- 3. LAYOUT ---
layout = dbc.Container([
    html.H1("Modélisation et Prédiction", className="text-center my-4 fw-bold", style={"color": "#004085"}),

    # LIGNE 1 : COMPARAISON & ROC
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Métriques de Performance", className="bg-primary text-white fw-bold"),
            dbc.CardBody([
                html.Table([
                    html.Thead(html.Tr([html.Th("Métrique"), html.Th("FDA"), html.Th("QDA")])),
                    html.Tbody([
                        html.Tr([html.Td("Précision (Acc)"), html.Td(f"{m_lda['acc']:.2%}"),
                                 html.Td(f"{m_qda['acc']:.2%}")]),
                        html.Tr([html.Td("Score F1", className="fw-bold text-primary"), html.Td(f"{m_lda['f1']:.2%}"),
                                 html.Td(f"{m_qda['f1']:.2%}")]),
                        html.Tr([html.Td("Rappel (Recall)"), html.Td(f"{m_lda['rec']:.2%}"),
                                 html.Td(f"{m_qda['rec']:.2%}")]),
                    ])
                ], className="table table-hover"),
                html.Div(dbc.Badge("Modèle séléctionné : FDA", color="success", className="p-2 w-100 mt-2"))
            ])
        ], style=style_card), width=4),

        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=create_roc_fig())), style=style_card), width=8)
    ], className="mb-4 d-flex align-items-stretch"),  # align-items-stretch harmonise la hauteur

    # LIGNE 2 : MATRICES de confusion
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=create_cm_fig(lda, "Confusion FDA (Stable)", "Blues"))),
                         style=style_card), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=create_cm_fig(qda, "Confusion QDA (Variable)", "Tealgrn"))),
                         style=style_card), width=6),
    ], className="mb-4"),

    # LIGNE 3 : SIMULATEUR
    dbc.Card([
        dbc.CardHeader("Simulateur de Décision de Crédit", className="bg-dark text-white fw-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Âge du client", className="fw-bold small"),
                    dcc.Input(id='p-age', type='number', value=35, className="form-control mb-3"),
                    html.Label("Revenu mensuel (FCFA)", className="fw-bold small"),
                    dcc.Input(id='p-rev', type='number', value=400000, className="form-control mb-3"),
                ], width=3),
                dbc.Col([
                    html.Label("Épargne totale (FCFA)", className="fw-bold small"),
                    dcc.Input(id='p-ep', type='number', value=200000, className="form-control mb-3"),
                    html.Label("Montant du prêt (FCFA)", className="fw-bold small"),
                    dcc.Input(id='p-mt', type='number', value=1500000, className="form-control mb-3"),
                ], width=3),
                dbc.Col([
                    html.Label("Endettement DSTI (%)", className="fw-bold small"),
                    dcc.Input(id='p-dsti', type='number', value=30, className="form-control mb-3"),
                    html.Label("Jours de retard (12m)", className="fw-bold small"),
                    dcc.Input(id='p-ret', type='number', value=0, className="form-control mb-3"),
                ], width=3),
                dbc.Col([
                    dbc.Button("Évaluer le dossier", id='btn-predict', color="primary",
                               className="w-100 h-100 fw-bold shadow-sm"),
                ], width=3, className="d-flex align-items-center"),
            ]),
            html.Hr(),
            html.Div(id='result-prediction', className="mt-3")
        ])
    ], style=style_card, className="mb-5")

], fluid=True, style={"backgroundColor": "#f0f4f8"})


# --- CALLBACK PRÉDICTION ---
@callback(
    Output('result-prediction', 'children'),
    Input('btn-predict', 'n_clicks'),
    [State('p-age', 'value'), State('p-rev', 'value'), State('p-ep', 'value'),
     State('p-ret', 'value'), State('p-dsti', 'value'), State('p-mt', 'value')]
)
def predict_score(n, age, rev, ep, ret, dsti, mt):
    if n:
        # Données formatées pour le modèle
        input_data = scaler.transform([[age, rev, ep, ret, dsti, mt]])
        prob = lda.predict_proba(input_data)[0][1]

        status = "ACCORDÉ" if prob < 0.5 else "REFUSÉ"
        color = "success" if prob < 0.5 else "danger"

        return dbc.Alert([
            html.H3(f"Résultat : {status}", className="text-center fw-bold"),
            html.P(f"Indice de risque calculé : {prob:.1%}", className="text-center"),
            dbc.Progress(value=prob * 100, color=color, className="mt-2", style={"height": "15px"})
        ], color=color, className="shadow-sm border-0")
    return html.Div("Veuillez saisir les informations du client.", className="text-center text-muted p-3")