"""
Seleção final de candidatos PROT (SLC6A7)
─────────────────────────────────────────
Entrada : candidates_inside_ad_chemberta.csv (262 compostos dentro do AD)
          prot_ligands.csv (50 compostos de referência)

Saída   : grupo_otimizacoes.csv   — 30 compostos mais próximos do AD
          grupo_novos_compostos.csv — 30 compostos mais distantes (ainda no AD)

Score composto (sem QSAR):
    score = (0.60 × norm_NLL_inv + 0.40 × norm_AD_inv) × penalidade_SAR

Filtro de novidade mínima: Tanimoto ECFP4 < 0.85 vs. qualquer composto do treino
(evita adições simples de CH₂ ou cópias quase idênticas do treino)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ── 0. Configurações ──────────────────────────────────────────────────────────
TANIMOTO_MAX     = 0.85   # novidade mínima: excluir se similar demais ao treino
N_SELECT         = 30     # candidatos por grupo
SAR_PENALTY      = 0.10   # penalidade multiplicativa por fragmento prejudicial

# Fragmentos prejudiciais identificados na análise BRICS do relatório
HARMFUL_SMARTS = [
    "[16*]c1ccccc1[16*]",   # bifenila não substituída  (Δ = −0.986)
    "[14*]c1ccccn1",        # piridina terminal          (Δ = −0.571)
    "[16*]c1ccccc1",        # fenila simples             (Δ = −0.541)
    "[16*]c1ccc(F)cc1",     # flúor para                 (Δ = −0.526)
    "[3*]O[3*]",            # éter/oxigênio ligante      (Δ = −0.456)
]

REF_CSV  = "D:\Arquivos_mestrado\Projetos_python\Triagem_PROT\LX6171_based\data\prot_ligands.csv"
CAND_CSV = "D:\Arquivos_mestrado\Projetos_python\Triagem_PROT\LX6171_based\data\candidates_inside_ad_chemberta.csv"

# ── 1. Carregar dados ─────────────────────────────────────────────────────────
df_ref  = pd.read_csv(REF_CSV)
df_cand = pd.read_csv(CAND_CSV).copy()

print(f"Candidatos no AD : {len(df_cand)}")
print(f"Referência       : {len(df_ref)}\n")

# ── 2. Calcular fingerprints ECFP4 ───────────────────────────────────────────
def to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

fps_ref  = [to_fp(s) for s in df_ref["SMILES"]]
fps_cand = [to_fp(s) for s in df_cand["SMILES"]]

# ── 3. Filtro de novidade mínima (Tanimoto < 0.85 vs. treino) ────────────────
def max_tanimoto_vs_ref(fp, fps_ref):
    if fp is None:
        return 1.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
    return max(sims)

print("Calculando similaridade Tanimoto vs. treino...")
df_cand["max_tanimoto"] = [
    max_tanimoto_vs_ref(fp, fps_ref) for fp in fps_cand
]

n_before = len(df_cand)
df_cand = df_cand[df_cand["max_tanimoto"] < TANIMOTO_MAX].reset_index(drop=True)
fps_cand = [to_fp(s) for s in df_cand["SMILES"]]  # atualizar fps

n_removed = n_before - len(df_cand)
print(f"Removidos por Tanimoto ≥ {TANIMOTO_MAX} : {n_removed}")
print(f"Candidatos restantes                    : {len(df_cand)}\n")

# ── 4. Penalidade SAR ─────────────────────────────────────────────────────────
# Os SMARTS com wildcards [n*] não são compatíveis com HasSubstructMatch diretamente.
# Removemos os wildcards para busca de subestrutura (conservador — pode haver
# falsos positivos, mas é seguro para penalização).
def clean_smarts(smarts):
    """Remove notações de wildcard [n*] para uso em HasSubstructMatch."""
    import re
    return re.sub(r'\[\d+\*\]', '[#6,#7,#8,#16,#9,#17,#35]', smarts)

harmful_mols = []
for s in HARMFUL_SMARTS:
    try:
        m = Chem.MolFromSmarts(clean_smarts(s))
        if m:
            harmful_mols.append(m)
    except Exception:
        pass

def count_harmful(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(mol.HasSubstructMatch(h) for h in harmful_mols)

df_cand["n_harmful"] = df_cand["SMILES"].apply(count_harmful)
df_cand["sar_penalty"] = df_cand["n_harmful"].apply(
    lambda n: max(0.50, 1.0 - n * SAR_PENALTY)
)

# ── 5. Score composto ─────────────────────────────────────────────────────────
# NLL invertido normalizado: menor NLL = maior confiança do REINVENT
nll_inv = 1 / (df_cand["NLL"] + 1e-6)
df_cand["norm_NLL_inv"] = (nll_inv - nll_inv.min()) / (nll_inv.max() - nll_inv.min())

# AD distance invertida normalizada: menor distância = mais dentro do AD
ad = df_cand["ad_distance_chemberta"]
ad_inv = 1 - ad
df_cand["norm_AD_inv"] = (ad_inv - ad_inv.min()) / (ad_inv.max() - ad_inv.min())

df_cand["score"] = (
    0.60 * df_cand["norm_NLL_inv"] +
    0.40 * df_cand["norm_AD_inv"]
) * df_cand["sar_penalty"]

df_cand = df_cand.sort_values("score", ascending=False).reset_index(drop=True)

print("── Score composto ───────────────────────────────────────")
print(f"  Score máx : {df_cand['score'].max():.4f}")
print(f"  Score mín : {df_cand['score'].min():.4f}")
print(f"  Score méd : {df_cand['score'].mean():.4f}\n")

# ── 6. Selecionar os dois grupos ──────────────────────────────────────────────
# Grupo 1: 30 maiores scores (mais próximos do AD, otimizações)
grupo_otim = df_cand.head(N_SELECT).copy()
grupo_otim.insert(0, "ID", [f"PROT_otim_{i+1:02d}" for i in range(len(grupo_otim))])

# Grupo 2: 30 menores scores (mais distantes, novos compostos)
grupo_novo = df_cand.tail(N_SELECT).copy()
grupo_novo = grupo_novo.sort_values("score", ascending=True).reset_index(drop=True)
grupo_novo.insert(0, "ID", [f"PROT_novo_{i+1:02d}" for i in range(len(grupo_novo))])

# ── 7. Exportar ───────────────────────────────────────────────────────────────
cols_out = ["ID", "SMILES", "NLL", "ad_distance_chemberta",
            "max_tanimoto", "n_harmful", "sar_penalty",
            "norm_NLL_inv", "norm_AD_inv", "score"]

grupo_otim[cols_out].to_csv("grupo_otimizacoes.csv", index=False)
grupo_novo[cols_out].to_csv("grupo_novos_compostos.csv", index=False)

# ── 8. Sumário ────────────────────────────────────────────────────────────────
print("── Grupo Otimizações (30 maiores scores) ────────────────")
print(f"  Score médio        : {grupo_otim['score'].mean():.4f}")
print(f"  AD distance média  : {grupo_otim['ad_distance_chemberta'].mean():.4f}")
print(f"  Tanimoto máx médio : {grupo_otim['max_tanimoto'].mean():.4f}")
print(f"  NLL médio          : {grupo_otim['NLL'].mean():.2f}")
print(f"  Com fragmentos SAR prejudiciais: {(grupo_otim['n_harmful'] > 0).sum()}")

print("\n── Grupo Novos Compostos (30 menores scores) ────────────")
print(f"  Score médio        : {grupo_novo['score'].mean():.4f}")
print(f"  AD distance média  : {grupo_novo['ad_distance_chemberta'].mean():.4f}")
print(f"  Tanimoto máx médio : {grupo_novo['max_tanimoto'].mean():.4f}")
print(f"  NLL médio          : {grupo_novo['NLL'].mean():.2f}")
print(f"  Com fragmentos SAR prejudiciais: {(grupo_novo['n_harmful'] > 0).sum()}")

print("\n✓ Arquivos gerados:")
print("  grupo_otimizacoes.csv")
print("  grupo_novos_compostos.csv")