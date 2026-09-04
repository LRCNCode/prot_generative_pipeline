"""
AD Pipeline: ChemBERTa Embeddings + kNN Distance
Dataset: PROT (SLC6A7) — 50 ligantes de referência + 744 moléculas geradas pelo REINVENT4

Substitui o AD anterior baseado em Tanimoto ECFP4 (threshold = 0,486, nn_similarity)
por um AD no espaço latente do ChemBERTa, capturando contexto químico mais rico.
"""

# ── Dependências ─────────────────────────────────────────────────────────────
# pip install transformers torch scikit-learn umap-learn rdkit matplotlib tqdm

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from rdkit import Chem

# ── 0. Configurações ─────────────────────────────────────────────────────────
MODEL_NAME  = "seyonec/ChemBERTa-zinc-base-v1"  # ChemBERTa pré-treinado no ZINC
K_NEIGHBORS = 5        # vizinhos para calcular d̄ (razoável para n=50)
PERCENTILE  = 95       # threshold = percentil 95 das distâncias intra-treino
BATCH_SIZE  = 32
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Caminhos de entrada
REF_CSV = "D:\Arquivos_mestrado\Projetos_python\Triagem_PROT\LX6171_based\data\prot_ligands.csv"
GEN_CSV = "D:\Arquivos_mestrado\Projetos_python\Triagem_PROT\LX6171_based\data\generated_prot.csv"
OUT_DIR = Path(".")

print(f"Dispositivo: {DEVICE}")
print(f"Modelo: {MODEL_NAME}")
print(f"k={K_NEIGHBORS} vizinhos | threshold = percentil {PERCENTILE}\n")

# ── 1. Carregar dados ─────────────────────────────────────────────────────────
df_ref = pd.read_csv(REF_CSV)   # colunas: SMILES, pChEMBL
df_gen = pd.read_csv(GEN_CSV)   # colunas: SMILES, SMILES_state, NLL

# O generated_prot.csv já tem apenas SMILES_state=1, mas validamos com RDKit
def validate_smiles(df, col="SMILES", label=""):
    valid_mask = df[col].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    n_inv = (~valid_mask).sum()
    if n_inv:
        print(f"  ⚠  {label}: {n_inv} SMILES inválidos removidos.")
    return df[valid_mask].reset_index(drop=True)

df_ref = validate_smiles(df_ref, label="referência")
df_gen = validate_smiles(df_gen, label="geradas")

print(f"Referência (treino) : {len(df_ref)} moléculas")
print(f"Geradas (REINVENT4) : {len(df_gen)} moléculas\n")

# ── 2. Embeddings ChemBERTa ───────────────────────────────────────────────────
print("Carregando modelo ChemBERTa...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_embeddings(smiles_list, batch_size=BATCH_SIZE, desc="Embeddings"):
    """
    Retorna embedding do token [CLS] para cada SMILES.
    O [CLS] agrega contexto global da sequência — análogo ao uso em
    classificação com BERT para sequências de linguagem natural.
    """
    all_emb = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc=desc):
        batch = smiles_list[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls)
    return np.vstack(all_emb)

emb_ref = get_embeddings(df_ref["SMILES"].tolist(), desc="Ref embeddings ")
emb_gen = get_embeddings(df_gen["SMILES"].tolist(), desc="Gen embeddings ")

# Normalização L2 — reduz viés de magnitude em alta dimensão (768-dim)
emb_ref_n = normalize(emb_ref)
emb_gen_n = normalize(emb_gen)

print(f"\nShape embeddings — ref: {emb_ref_n.shape} | gen: {emb_gen_n.shape}")

# ── 3. Definir AD via kNN ─────────────────────────────────────────────────────
print(f"\nFitando kNN (k={K_NEIGHBORS}) no espaço ChemBERTa...")
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean", n_jobs=-1)
knn.fit(emb_ref_n)

# Distâncias intra-referência (leave-one-in: cada ponto consulta os k vizinhos
# dentro do próprio conjunto de treino, incluindo a si mesmo com d=0 para k=1)
# Usamos k+1 para excluir a auto-distância (d=0) no cálculo do threshold
knn_cal = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="euclidean", n_jobs=-1)
knn_cal.fit(emb_ref_n)
dist_ref_raw, _ = knn_cal.kneighbors(emb_ref_n)
# Remove a coluna 0 (auto-distância = 0)
dist_ref   = dist_ref_raw[:, 1:]
d_mean_ref = dist_ref.mean(axis=1)

# Threshold: percentil 95 das distâncias médias intra-treino
threshold = np.percentile(d_mean_ref, PERCENTILE)
print(f"Threshold AD (percentil {PERCENTILE} intra-treino): {threshold:.4f}")
print(f"  d̄ ref — min: {d_mean_ref.min():.4f} | média: {d_mean_ref.mean():.4f} | max: {d_mean_ref.max():.4f}")

# Distâncias das moléculas geradas ao espaço de referência
dist_gen, _  = knn.kneighbors(emb_gen_n)
d_mean_gen   = dist_gen.mean(axis=1)

# ── 4. Classificar e exportar ─────────────────────────────────────────────────
df_gen["ad_distance_chemberta"] = np.round(d_mean_gen, 6)
df_gen["inside_ad_chemberta"]   = d_mean_gen <= threshold

n_in  = df_gen["inside_ad_chemberta"].sum()
n_out = len(df_gen) - n_in
pct   = 100 * n_in / len(df_gen)

print(f"\n{'─'*45}")
print(f"Dentro do AD : {n_in:>4}  ({pct:.1f}%)")
print(f"Fora do AD   : {n_out:>4}  ({100-pct:.1f}%)")
print(f"{'─'*45}")

# Exportar todos com a coluna de AD
out_all = OUT_DIR / "generated_prot_with_ad_chemberta.csv"
df_gen.to_csv(out_all, index=False)
print(f"\nCSV completo salvo em: {out_all}")

# Exportar apenas os dentro do AD
out_in = OUT_DIR / "candidates_inside_ad_chemberta.csv"
df_gen[df_gen["inside_ad_chemberta"]].to_csv(out_in, index=False)
print(f"Candidatos dentro do AD: {out_in}")

# ── 5. Visualização ───────────────────────────────────────────────────────────
print("\nGerando visualizações...")

# PCA 2D (determinístico, mais rápido que UMAP para relatório)
pca = PCA(n_components=2, random_state=42)
all_emb_n = np.vstack([emb_ref_n, emb_gen_n])
proj      = pca.fit_transform(all_emb_n)

proj_ref = proj[:len(emb_ref_n)]
proj_gen = proj[len(emb_ref_n):]

mask_in  = df_gen["inside_ad_chemberta"].values
mask_out = ~mask_in

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Applicability Domain — ChemBERTa + kNN | PROT (SLC6A7)", fontsize=13, fontweight="bold")

# ── Painel esquerdo: PCA ──
ax = axes[0]
ax.scatter(*proj_gen[mask_out].T, c="#f28b82", s=18, alpha=0.45, label=f"Fora do AD (n={mask_out.sum()})", zorder=1)
ax.scatter(*proj_gen[mask_in].T,  c="#34a853", s=22, alpha=0.65, label=f"Dentro do AD (n={mask_in.sum()})", zorder=2)
ax.scatter(*proj_ref.T,           c="#1a73e8", s=55, alpha=0.90, edgecolors="white", linewidths=0.5,
           label=f"Referência PROT (n={len(df_ref)})", zorder=3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Espaço ChemBERTa (PCA 2D)")
ax.legend(fontsize=9, framealpha=0.85)

# ── Painel direito: distribuição de distâncias ──
ax2 = axes[1]
ax2.hist(d_mean_ref, bins=15, color="#1a73e8", alpha=0.75, label="Referência (d̄ intra-treino)", density=True)
ax2.hist(d_mean_gen[mask_in],  bins=30, color="#34a853", alpha=0.65, label="Geradas – dentro AD", density=True)
ax2.hist(d_mean_gen[mask_out], bins=30, color="#f28b82", alpha=0.50, label="Geradas – fora AD", density=True)
ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
ax2.set_xlabel("d̄ média aos k vizinhos mais próximos")
ax2.set_ylabel("Densidade")
ax2.set_title("Distribuição de Distâncias no Espaço Latente")
ax2.legend(fontsize=9, framealpha=0.85)

plt.tight_layout()
fig_path = OUT_DIR / "ad_chemberta_prot.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura salva em: {fig_path}")

# ── 6. Comparação com AD anterior (Tanimoto ECFP4) ───────────────────────────
# O relatório documenta: 127 candidatos dentro do AD anterior (Tanimoto ≥ 0,486)
print("\n── Comparação AD anterior vs. atual ──────────────────────")
print(f"  AD anterior (Tanimoto ECFP4, nn_sim ≥ 0.486) : 127 / 618  (~20.5%)")
print(f"  AD atual (ChemBERTa kNN, d̄ ≤ {threshold:.3f})      : {n_in} / {len(df_gen)} ({pct:.1f}%)")
print(f"\n  Obs.: o denominador mudou de 618 (pós-filtro Lipinski/PAINS)")
print(f"        para {len(df_gen)} (total válido pelo REINVENT), pois o AD ChemBERTa")
print(f"        pode ser aplicado antes ou depois do funil de drug-likeness.")

# ── 7. Estatísticas descritivas dos candidatos dentro do AD ───────────────────
df_in = df_gen[df_gen["inside_ad_chemberta"]].copy()
print(f"\n── Estatísticas dos candidatos dentro do AD ──────────────")
print(f"  NLL médio  : {df_in['NLL'].mean():.2f}  (referência total: {df_gen['NLL'].mean():.2f})")
print(f"  NLL mínimo : {df_in['NLL'].min():.2f}")
print(f"  d̄ mínima  : {df_in['ad_distance_chemberta'].min():.4f}")
print(f"  d̄ média   : {df_in['ad_distance_chemberta'].mean():.4f}")
print(f"  d̄ máxima  : {df_in['ad_distance_chemberta'].max():.4f}")

print("\n✓ Pipeline concluído.")
