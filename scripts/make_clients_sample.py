from pathlib import Path
import sys
import pandas as pd

# --- Permet d'importer src/* même si on lance le script depuis un autre dossier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main(n_rows: int = 10000, random_state: int = 42) -> None:
    # 1) Chemins
    raw_path = RAW_DATA_DIR / "application_train.csv"
    out_path = PROCESSED_DATA_DIR / "clients_sample.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Charger le CSV brut
    df = pd.read_csv(raw_path)

    # 3) Retirer TARGET (l'API ne reçoit pas la vérité terrain)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    # 4) Vérifier que l'ID client existe
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("La colonne 'SK_ID_CURR' est introuvable dans application_train.csv")

    # 5) Mettre SK_ID_CURR en première colonne (plus pratique pour les lookups)
    cols = ["SK_ID_CURR"] + [c for c in df.columns if c != "SK_ID_CURR"]
    df = df[cols]

    # 6) Prendre un échantillon (démo)
    n = min(n_rows, len(df))
    sample = df.sample(n=n, random_state=random_state)

    # 7) Sauvegarder en parquet
    sample.to_parquet(out_path, index=False)

    print("✅ clients_sample.parquet créé !")
    print("   Chemin :", out_path)
    print("   Shape  :", sample.shape)
    print("   Colonnes :", len(sample.columns))


if __name__ == "__main__":
    main()