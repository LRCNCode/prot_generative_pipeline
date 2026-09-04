"""
Scaffold consenso — PROT (SLC6A7)
===================================
Abordagem combinada:
  1. Murcko Scaffold mais frequente → scaffold base
  2. MCS (Maximum Common Substructure) entre todos os compostos
     → identifica átomos verdadeiramente conservados
  3. Sobreposição Murcko ∩ MCS → regiões conservadas (azul)
  4. Átomos do Murcko fora do MCS → pontos de diversificação (vermelho)

Outputs:
  scaffold_annotated.png     — imagem anotada (conservado=azul, diversif.=vermelho)
  scaffold_clean.png         — scaffold limpo sem anotações
  scaffold_murcko.smi        — SMILES do scaffold Murcko dominante
  scaffold_mcs.smi           — SMARTS do MCS
  scaffold_report.txt        — relatório textual
"""

import os
import io
import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFMCS import FindMCS, MCSParameters, AtomCompare, BondCompare
from rdkit.Chem.Draw import rdMolDraw2D

OUT_DIR      = "scaffold_output/"
REFERENCE_CSV = "D:/Arquivos_mestrado/Projetos_python/Triagem_PROT/LX6171_based/data/prot_ligands.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_reference(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "smiles" in cl:
            col_map[c] = "SMILES"
        elif any(k in cl for k in ["pchembl","activity","pic50","pki"]):
            col_map[c] = "pActivity"
    df.rename(columns=col_map, inplace=True)
    valid = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]).strip())
        if mol:
            valid.append({
                "SMILES":    Chem.MolToSmiles(mol),
                "pActivity": float(row["pActivity"]),
                "mol":       mol,
            })
    result = pd.DataFrame(valid)
    print(f"  Compostos carregados: {len(result)}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 2. MURCKO — scaffold mais frequente
# ─────────────────────────────────────────────────────────────────────────────

def get_murcko_scaffolds(df):
    """
    Para cada composto, extrai o Murcko Scaffold.
    Retorna DataFrame com scaffolds, frequência e pActivity média.
    """
    scaffold_data = defaultdict(list)
    scaffold_mols = {}

    for _, row in df.iterrows():
        try:
            sc_mol = MurckoScaffold.GetScaffoldForMol(row["mol"])
            sc_smi = Chem.MolToSmiles(sc_mol)
            scaffold_data[sc_smi].append(row["pActivity"])
            if sc_smi not in scaffold_mols:
                scaffold_mols[sc_smi] = sc_mol
        except Exception:
            pass

    rows = []
    for smi, acts in scaffold_data.items():
        rows.append({
            "scaffold_smiles": smi,
            "frequency":       len(acts),
            "mean_pActivity":  round(np.mean(acts), 3),
            "mol":             scaffold_mols[smi],
        })

    sc_df = (pd.DataFrame(rows)
             .sort_values("frequency", ascending=False)
             .reset_index(drop=True))

    print(f"  Scaffolds únicos encontrados: {len(sc_df)}")
    print(f"  Scaffold dominante: {sc_df.iloc[0]['scaffold_smiles']}")
    print(f"    Frequência: {sc_df.iloc[0]['frequency']} compostos "
          f"({100*sc_df.iloc[0]['frequency']/len(df):.0f}% do dataset)")
    print(f"    pActivity média: {sc_df.iloc[0]['mean_pActivity']}")

    return sc_df

# ─────────────────────────────────────────────────────────────────────────────
# 3. MCS — subestrutura máxima comum
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcs(df, timeout=60):
    """
    Calcula o MCS entre todos os compostos do dataset.
    Usa comparação de átomos por elemento e ligações por tipo.
    Retorna o objeto MCS e o mol do MCS.
    """
    mols = df["mol"].tolist()

    params = MCSParameters()
    params.AtomTyper    = AtomCompare.CompareElements
    params.BondTyper    = BondCompare.CompareOrder
    params.Timeout      = timeout
    params.Threshold    = 0.8   # presença em ≥ 80% dos compostos

    print(f"  Calculando MCS entre {len(mols)} compostos (timeout={timeout}s)...")
    mcs_result = FindMCS(mols, params)

    print(f"  MCS: {mcs_result.numAtoms} átomos, {mcs_result.numBonds} ligações")
    print(f"  SMARTS: {mcs_result.smartsString}")

    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    return mcs_result, mcs_mol

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANOTAÇÃO — conservado vs. diversificação
# ─────────────────────────────────────────────────────────────────────────────

def annotate_scaffold(murcko_mol, mcs_mol):
    """
    Mapeia os átomos do Murcko scaffold em relação ao MCS:
      - Átomos que fazem match com o MCS → conservados (azul)
      - Átomos do Murcko fora do MCS     → pontos de diversificação (vermelho)

    Retorna:
      conserved_atoms   — índices dos átomos conservados no Murcko
      diverse_atoms     — índices dos pontos de diversificação
      conserved_bonds   — índices das ligações conservadas
      diverse_bonds     — índices das ligações de diversificação
    """
    match = murcko_mol.GetSubstructMatch(mcs_mol)
    conserved_atoms = set(match)
    diverse_atoms   = set(range(murcko_mol.GetNumAtoms())) - conserved_atoms

    conserved_bonds, diverse_bonds = set(), set()
    for bond in murcko_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in conserved_atoms and a2 in conserved_atoms:
            conserved_bonds.add(bond.GetIdx())
        else:
            diverse_bonds.add(bond.GetIdx())

    print(f"  Átomos conservados (MCS ∩ Murcko): {len(conserved_atoms)}")
    print(f"  Pontos de diversificação:           {len(diverse_atoms)}")

    return (list(conserved_atoms), list(diverse_atoms),
            list(conserved_bonds), list(diverse_bonds))


def render_annotated(murcko_mol, conserved_atoms, diverse_atoms,
                     conserved_bonds, diverse_bonds,
                     size=(800, 600)):
    """
    Renderiza o scaffold com:
      Azul  (#3266AD) → regiões conservadas pelo MCS
      Vermelho (#D85A30) → pontos de diversificação
    """
    BLUE = (0.196, 0.400, 0.678)
    RED  = (0.851, 0.353, 0.188)

    atom_colors = {}
    bond_colors = {}

    for idx in conserved_atoms:
        atom_colors[idx] = BLUE
    for idx in diverse_atoms:
        atom_colors[idx] = RED
    for idx in conserved_bonds:
        bond_colors[idx] = BLUE
    for idx in diverse_bonds:
        bond_colors[idx] = RED

    all_atoms = conserved_atoms + diverse_atoms
    all_bonds = conserved_bonds + diverse_bonds

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation  = False
    opts.addAtomIndices        = False
    opts.bondLineWidth         = 2.5
    opts.highlightRadius       = 0.35

    drawer.DrawMolecule(
        murcko_mol,
        highlightAtoms=all_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=all_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def svg_to_png(svg_string, out_path):
    """Converte SVG para PNG via cairosvg ou salva SVG direto."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_string.encode(), write_to=out_path)
        print(f"  PNG salvo: {out_path}")
    except ImportError:
        # salva SVG
        svg_path = out_path.replace(".png", ".svg")
        with open(svg_path, "w") as f:
            f.write(svg_string)
        print(f"  cairosvg não disponível — SVG salvo: {svg_path}")


def render_clean(murcko_mol, size=(800, 600)):
    """Scaffold limpo sem anotações."""
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    opts.addAtomIndices       = False
    opts.bondLineWidth        = 2.5
    drawer.DrawMolecule(murcko_mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISE ADICIONAL — variabilidade por posição
# ─────────────────────────────────────────────────────────────────────────────

def analyze_substitution_points(df, murcko_dominant_smi, diverse_atoms_idx, murcko_mol):
    """
    Para cada ponto de diversificação no scaffold, conta quantos
    compostos do dataset têm substituições diferentes ali e qual o
    impacto médio na atividade.
    """
    rows = []
    for atom_idx in diverse_atoms_idx:
        atom = murcko_mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()

        # conta compostos que têm match no scaffold e variam nessa posição
        n_match = 0
        acts = []
        for _, row in df.iterrows():
            match = row["mol"].GetSubstructMatch(murcko_mol)
            if match:
                n_match += 1
                acts.append(row["pActivity"])

        rows.append({
            "atom_idx":    atom_idx,
            "atom_symbol": symbol,
            "n_compounds": n_match,
            "mean_pAct":   round(np.mean(acts), 3) if acts else None,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 6. RELATÓRIO TEXTUAL
# ─────────────────────────────────────────────────────────────────────────────

def write_report(sc_df, mcs_result, conserved_atoms, diverse_atoms, df, out_path):
    dominant = sc_df.iloc[0]
    lines = [
        "=" * 60,
        " RELATÓRIO DE SCAFFOLD — PROT (SLC6A7)",
        "=" * 60,
        "",
        "1. SCAFFOLD MURCKO DOMINANTE",
        "-" * 40,
        f"  SMILES:      {dominant['scaffold_smiles']}",
        f"  Frequência:  {dominant['frequency']} compostos "
        f"({100*dominant['frequency']/len(df):.0f}% do dataset)",
        f"  pActivity média: {dominant['mean_pActivity']}",
        "",
        "2. TODOS OS SCAFFOLDS IDENTIFICADOS",
        "-" * 40,
    ]
    for _, row in sc_df.iterrows():
        lines.append(f"  [{row['frequency']:>2}x]  pAct={row['mean_pActivity']:.3f}  "
                     f"{row['scaffold_smiles']}")
    lines += [
        "",
        "3. MAXIMUM COMMON SUBSTRUCTURE (MCS)",
        "-" * 40,
        f"  SMARTS:    {mcs_result.smartsString}",
        f"  Átomos:    {mcs_result.numAtoms}",
        f"  Ligações:  {mcs_result.numBonds}",
        "",
        "4. ANOTAÇÃO DO SCAFFOLD",
        "-" * 40,
        f"  Regiões conservadas (azul):        {len(conserved_atoms)} átomos",
        f"  Pontos de diversificação (vermelho): {len(diverse_atoms)} átomos",
        "",
        "5. INTERPRETAÇÃO",
        "-" * 40,
        "  As regiões conservadas (azul) representam o núcleo",
        "  farmacofórico essencial — modificações nessa região",
        "  provavelmente afetam negativamente a atividade.",
        "",
        "  Os pontos de diversificação (vermelho) são as posições",
        "  onde o dataset já explora variações estruturais.",
        "  São os sítios prioritários para modificações racionais",
        "  em campanhas de otimização de leads.",
        "",
        "  Outputs:",
        "    scaffold_annotated.png  — imagem anotada",
        "    scaffold_clean.png      — scaffold limpo",
        "    scaffold_murcko.smi     — SMILES do Murcko dominante",
        "    scaffold_mcs.smi        — SMARTS do MCS",
        "=" * 60,
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Relatório salvo: {out_path}")
    # também imprime no console
    print()
    for line in lines:
        print(line)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(" SCAFFOLD CONSENSO — PROT (SLC6A7)")
    print("="*60)

    # 1. Carrega
    print("\n[1] Carregando dataset...")
    df = load_reference(REFERENCE_CSV)

    # 2. Murcko
    print("\n[2] Análise de scaffolds Murcko...")
    sc_df = get_murcko_scaffolds(df)
    murcko_mol = sc_df.iloc[0]["mol"]
    murcko_smi = sc_df.iloc[0]["scaffold_smiles"]

    # salva SMILES do scaffold dominante
    with open(OUT_DIR + "scaffold_murcko.smi", "w") as f:
        f.write(murcko_smi + "\n")

    # 3. MCS
    print("\n[3] Calculando MCS...")
    mcs_result, mcs_mol = compute_mcs(df)

    # salva SMARTS do MCS
    with open(OUT_DIR + "scaffold_mcs.smi", "w") as f:
        f.write(mcs_result.smartsString + "\n")

    # 4. Anotação
    print("\n[4] Anotando scaffold (conservado vs. diversificação)...")
    if mcs_mol is None or not murcko_mol.HasSubstructMatch(mcs_mol):
        print("  AVISO: MCS não faz match no Murcko — usando MCS direto como scaffold")
        # fallback: usa o MCS como scaffold base
        murcko_mol = Chem.RWMol(mcs_mol)
        Chem.SanitizeMol(murcko_mol)
        conserved_atoms = list(range(murcko_mol.GetNumAtoms()))
        diverse_atoms   = []
        conserved_bonds = list(range(murcko_mol.GetNumBonds()))
        diverse_bonds   = []
    else:
        conserved_atoms, diverse_atoms, conserved_bonds, diverse_bonds = \
            annotate_scaffold(murcko_mol, mcs_mol)

    # 5. Renderiza scaffold anotado
    print("\n[5] Renderizando imagens...")
    svg_annotated = render_annotated(
        murcko_mol, conserved_atoms, diverse_atoms,
        conserved_bonds, diverse_bonds, size=(900, 650)
    )
    svg_to_png(svg_annotated, OUT_DIR + "scaffold_annotated.png")

    # scaffold limpo
    svg_clean = render_clean(murcko_mol, size=(900, 650))
    svg_to_png(svg_clean, OUT_DIR + "scaffold_clean.png")

    # 6. Relatório
    print("\n[6] Gerando relatório...")
    write_report(sc_df, mcs_result, conserved_atoms, diverse_atoms,
                 df, OUT_DIR + "scaffold_report.txt")

    print("\n" + "="*60)
    print(" CONCLUÍDO")
    print("="*60)
    print(f"  Scaffold Murcko dominante: {murcko_smi}")
    print(f"  Átomos conservados:  {len(conserved_atoms)}")
    print(f"  Pontos de diversif.: {len(diverse_atoms)}")
    print(f"  Outputs em: {OUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
