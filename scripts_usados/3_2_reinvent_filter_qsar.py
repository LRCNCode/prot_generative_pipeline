"""
Post-processing pipeline: REINVENT output → filtros → QSAR → ranking
=====================================================================
Entrada : generated_prot.csv  (saída do REINVENT4)
Referência: prot_ligands.csv  (50 compostos do dataset original)

Etapas
------
1. Carrega e valida SMILES (RDKit)
2. Remove duplicatas internas e moléculas já presentes no dataset de treino
3. Filtragem por drug-likeness (Lipinski + PAINS + tamanho mínimo)
4. Calcula descritores 2D e NLL do REINVENT
5. Applicability Domain (Tanimoto nn ao dataset de treino)
6. Modelo QSAR (Random Forest + Ridge + SVR → consenso)
7. Score final composto: combina pActivity predita, NLL, AD e penalidade SAR
8. Exporta resultados ranqueados
"""

import pandas as pd
import numpy as np
import warnings
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, AllChem, rdMolDescriptors,
    FilterCatalog, rdFingerprintGenerator
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem.BRICS import BRICSDecompose

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES — edite aqui se necessário
# ─────────────────────────────────────────────────────────────────────────────
GENERATED_CSV  = "/mnt/user-data/uploads/generated_prot.csv"
REFERENCE_CSV  = "/mnt/user-data/uploads/prot_ligands.csv"
OUTPUT_CSV     = "/mnt/user-data/outputs/reinvent_ranked_candidates.csv"
OUTPUT_TOP_CSV = "/mnt/user-data/outputs/reinvent_top50_candidates.csv"

MW_MAX    = 600
LOGP_MAX  = 6.0
HBD_MAX   = 5
HBA_MAX   = 12
MIN_ATOMS = 15

AD_THRESHOLD = None  # None = calibrar automaticamente (percentil 5)

W_PACTIVITY = 0.60
W_NLL       = 0.20
W_AD        = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_generated(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    valid = []
    for _, row in df.iterrows():
        smi = str(row["SMILES"]).strip()
        if int(row.get("SMILES_state", 1)) != 1:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        valid.append({
            "SMILES_original": smi,
            "SMILES":          canon,
            "NLL":             float(row["NLL"]),
            "mol":             mol,
        })

    result = pd.DataFrame(valid).drop_duplicates(subset="SMILES").reset_index(drop=True)
    print(f"  Compostos gerados válidos (pós-dedup): {len(result)}")
    return result


def load_reference(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "smiles" in cl:
            col_map[c] = "SMILES"
        elif any(k in cl for k in ["pchembl", "activity", "pic50", "pki"]):
            col_map[c] = "pActivity"
    df.rename(columns=col_map, inplace=True)

    valid = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]).strip())
        if mol is None:
            continue
        valid.append({
            "SMILES":    Chem.MolToSmiles(mol),
            "pActivity": float(row["pActivity"]),
            "mol":       mol,
        })

    result = pd.DataFrame(valid)
    print(f"  Compostos de referência válidos: {len(result)}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 2. REMOÇÃO DE SOBREPOSIÇÃO COM O DATASET DE TREINO
# ─────────────────────────────────────────────────────────────────────────────

def remove_known(gen_df, ref_df):
    known  = set(ref_df["SMILES"].tolist())
    before = len(gen_df)
    gen_df = gen_df[~gen_df["SMILES"].isin(known)].reset_index(drop=True)
    print(f"  Removidos {before - len(gen_df)} compostos já presentes no dataset de treino")
    print(f"  Restantes: {len(gen_df)}")
    return gen_df

# ─────────────────────────────────────────────────────────────────────────────
# 3. DESCRITORES E FILTROS
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptors(mol):
    return {
        "MW":           round(Descriptors.MolWt(mol), 2),
        "LogP":         round(Descriptors.MolLogP(mol), 3),
        "TPSA":         round(Descriptors.TPSA(mol), 2),
        "HBD":          rdMolDescriptors.CalcNumHBD(mol),
        "HBA":          rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds":     rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ArRings":      rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms":   mol.GetNumHeavyAtoms(),
        "QED":          round(Descriptors.qed(mol), 4),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }


def apply_filters(gen_df):
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog.FilterCatalog(pains_params)

    rows = []
    reasons_fail = defaultdict(int)

    for _, row in gen_df.iterrows():
        mol  = row["mol"]
        desc = compute_descriptors(mol)

        if desc["HeavyAtoms"] < MIN_ATOMS:
            reasons_fail["tamanho_minimo"] += 1
            continue
        if desc["MW"]   > MW_MAX:   reasons_fail["MW"] += 1;   continue
        if desc["LogP"] > LOGP_MAX: reasons_fail["LogP"] += 1; continue
        if desc["HBD"]  > HBD_MAX:  reasons_fail["HBD"] += 1;  continue
        if desc["HBA"]  > HBA_MAX:  reasons_fail["HBA"] += 1;  continue
        if pains_catalog.HasMatch(mol):
            reasons_fail["PAINS"] += 1
            continue

        rows.append({**row.to_dict(), **desc})

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  Compostos após filtros: {len(result)}")
    print("  Motivos de exclusão:")
    for reason, count in sorted(reasons_fail.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 4. FINGERPRINTS
# ─────────────────────────────────────────────────────────────────────────────

def get_fpgen():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def mols_to_fp_matrix(mols, fpgen):
    return np.array([fpgen.GetFingerprintAsNumPy(m) for m in mols])

# ─────────────────────────────────────────────────────────────────────────────
# 5. APPLICABILITY DOMAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_ad(ref_df, fpgen, threshold=None):
    train_fps = [fpgen.GetFingerprint(m) for m in ref_df["mol"]]

    if threshold is None:
        nn_sims = []
        for i, fp_i in enumerate(train_fps):
            sims = [DataStructs.TanimotoSimilarity(fp_i, fp_j)
                    for j, fp_j in enumerate(train_fps) if i != j]
            nn_sims.append(max(sims))
        threshold = float(np.percentile(nn_sims, 5))

    print(f"  AD threshold (p5 nn intra-treino): {threshold:.3f}")

    def predict_ad(mols):
        results = []
        for mol in mols:
            fp   = fpgen.GetFingerprint(mol)
            sims = [DataStructs.TanimotoSimilarity(fp, fp_t) for fp_t in train_fps]
            nn   = max(sims)
            results.append({
                "nn_similarity": round(nn, 4),
                "in_AD":         bool(nn >= threshold),
            })
        return pd.DataFrame(results)

    return predict_ad, threshold

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELO QSAR (ensemble RF + Ridge + SVR)
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble_qsar(ref_df, fpgen):
    X = mols_to_fp_matrix(ref_df["mol"].tolist(), fpgen)
    y = ref_df["pActivity"].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    models = {
        "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=42,
                                               n_jobs=-1), X),
        "Ridge":        (Ridge(alpha=1.0),                                    X_sc),
        "SVR":          (SVR(kernel="rbf", C=1.0, gamma="scale"),             X_sc),
    }

    trained = {}
    print("  R² k-fold (k=5) por modelo:")
    for name, (model, X_in) in models.items():
        scores = cross_val_score(model, X_in, y, cv=kf, scoring="r2")
        print(f"    {name:<15}: {scores.mean():.3f} ± {scores.std():.3f}")
        model.fit(X_in, y)
        trained[name] = model

    def predict_consensus(X_new):
        preds = []
        preds.append(trained["RandomForest"].predict(X_new))
        X_new_sc = scaler.transform(X_new)
        preds.append(trained["Ridge"].predict(X_new_sc))
        preds.append(trained["SVR"].predict(X_new_sc))
        return np.mean(preds, axis=0)

    return predict_consensus

# ─────────────────────────────────────────────────────────────────────────────
# 7. PENALIDADE SAR
# ─────────────────────────────────────────────────────────────────────────────

def get_harmful_fragments(ref_df, delta_threshold=-0.3, min_freq=2):
    mean_global = ref_df["pActivity"].mean()
    frag_data   = defaultdict(list)

    for _, row in ref_df.iterrows():
        try:
            for fsmi in BRICSDecompose(row["mol"]):
                fmol = Chem.MolFromSmiles(fsmi)
                if fmol:
                    frag_data[Chem.MolToSmiles(fmol)].append(row["pActivity"])
        except Exception:
            pass

    harmful = []
    for smi, acts in frag_data.items():
        if len(acts) < min_freq:
            continue
        if np.mean(acts) - mean_global < delta_threshold:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                harmful.append(mol)

    print(f"  Fragmentos prejudiciais identificados: {len(harmful)}")
    return harmful


def count_harmful_fragments(mol, harmful_frags):
    return sum(1 for frag in harmful_frags if mol.HasSubstructMatch(frag))

# ─────────────────────────────────────────────────────────────────────────────
# 8. SCORE FINAL COMPOSTO
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(df):
    def minmax(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    norm_pact = minmax(df["predicted_pActivity"])
    norm_nll  = minmax(1.0 / df["NLL"])
    norm_ad   = minmax(df["nn_similarity"])

    raw_score = (W_PACTIVITY * norm_pact
               + W_NLL       * norm_nll
               + W_AD        * norm_ad)

    penalty = df["n_harmful_frags"].apply(lambda n: max(0.5, 1.0 - 0.10 * n))
    df["composite_score"] = (raw_score * penalty).round(4)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fpgen = get_fpgen()

    print("\n" + "="*60)
    print(" REINVENT POST-PROCESSING PIPELINE — PROT (SLC6A7)")
    print("="*60)

    print("\n[1] Carregando arquivos...")
    gen_df = load_generated(GENERATED_CSV)
    ref_df = load_reference(REFERENCE_CSV)

    print("\n[2] Removendo compostos já conhecidos...")
    gen_df = remove_known(gen_df, ref_df)

    print("\n[3] Aplicando filtros (Lipinski + PAINS + tamanho)...")
    gen_df = apply_filters(gen_df)

    if len(gen_df) == 0:
        print("ERRO: nenhum composto passou pelos filtros.")
        return

    print("\n[4] Gerando fingerprints ECFP4...")
    X_gen = mols_to_fp_matrix(gen_df["mol"].tolist(), fpgen)
    print(f"  Shape X_gen: {X_gen.shape}")

    print("\n[5] Calibrando Applicability Domain...")
    predict_ad, ad_threshold = build_ad(ref_df, fpgen, threshold=AD_THRESHOLD)
    ad_results = predict_ad(gen_df["mol"].tolist())
    gen_df["nn_similarity"] = ad_results["nn_similarity"].values
    gen_df["in_AD"]         = ad_results["in_AD"].values
    n_in_ad = gen_df["in_AD"].sum()
    print(f"  Candidatos dentro do AD: {n_in_ad}/{len(gen_df)} "
          f"({100*n_in_ad/len(gen_df):.1f}%)")

    print("\n[6] Treinando ensemble QSAR no dataset de referência...")
    predict_fn = train_ensemble_qsar(ref_df, fpgen)
    gen_df["predicted_pActivity"] = np.round(predict_fn(X_gen), 3)

    print("\n[7] Identificando fragmentos prejudiciais (SAR)...")
    harmful_frags = get_harmful_fragments(ref_df)
    gen_df["n_harmful_frags"] = [
        count_harmful_fragments(m, harmful_frags) for m in gen_df["mol"]
    ]

    print("\n[8] Calculando score composto final...")
    gen_df = compute_composite_score(gen_df)

    gen_df = gen_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    gen_df["rank"] = gen_df.index + 1

    output_cols = [
        "rank", "SMILES", "composite_score",
        "predicted_pActivity", "NLL", "nn_similarity", "in_AD",
        "n_harmful_frags", "MW", "LogP", "TPSA", "HBD", "HBA",
        "RotBonds", "ArRings", "QED", "FractionCSP3", "HeavyAtoms",
    ]
    out_df = gen_df[output_cols]

    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df[out_df["in_AD"]].head(50).to_csv(OUTPUT_TOP_CSV, index=False)

    print("\n" + "="*60)
    print(" RESUMO")
    print("="*60)
    print(f"  Compostos gerados (entrada):        744")
    print(f"  Após remover duplicatas/conhecidos: {len(gen_df)}")
    print(f"  Dentro do AD:                       {int(gen_df['in_AD'].sum())}")
    print(f"  Sem fragmentos prejudiciais:        {int((gen_df['n_harmful_frags']==0).sum())}")
    print(f"  pActivity predita — média:          {gen_df['predicted_pActivity'].mean():.3f}")
    print(f"  pActivity predita — max:            {gen_df['predicted_pActivity'].max():.3f}")
    print()
    print("  Top 5 candidatos (in AD):")
    top5 = out_df[out_df["in_AD"]].head(5)
    for _, r in top5.iterrows():
        print(f"    #{int(r['rank'])}  score={r['composite_score']:.4f}  "
              f"pAct={r['predicted_pActivity']:.3f}  "
              f"NLL={r['NLL']:.1f}  "
              f"nn={r['nn_similarity']:.3f}  "
              f"MW={r['MW']:.0f}  QED={r['QED']:.3f}")
        print(f"       {r['SMILES']}")
    print()
    print(f"  Arquivos salvos:")
    print(f"    {OUTPUT_CSV}  ({len(out_df)} compostos)")
    print(f"    {OUTPUT_TOP_CSV}  (top 50 in AD)")
    print("="*60)


if __name__ == "__main__":
    main()
