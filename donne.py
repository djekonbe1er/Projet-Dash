import pandas as pd
import os
import sys


def load_data():
    if getattr(sys, 'frozen', False):
        # En mode EXE : le fichier est à la racine de _MEIPASS
        base_path = sys._MEIPASS
    else:
        # En mode SCRIPT : on remonte d'un niveau car donne.py est dans 'utils'
        # On passe de .../Ndoasnan_Armand_D/utils/ à .../Ndoasnan_Armand_D/
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_path = os.path.join(base_path, "microfinance_credit_risk.xlsx")

    try:
        df = pd.read_excel(file_path)
        if df is None:
            raise ValueError("Le DataFrame est vide")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        # On affiche le chemin tenté pour t'aider à débugger
        print(f"Chemin tenté : {file_path}")
        return pd.DataFrame()  # Retourne un DataFrame vide pour éviter le plantage 'NoneType'