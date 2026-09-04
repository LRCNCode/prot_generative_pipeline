[README.md](https://github.com/user-attachments/files/31853530/README.md)
# Pipeline Generativo de Descoberta de Inibidores — PROT (SLC6A7)

Pipeline computacional para geração e triagem de candidatos a inibidores do transportador de prolina PROT (SLC6A7), desenvolvido como parte de um projeto de mestrado em neurofarmacologia computacional (UFG) / NeuroSpin.

Combina transfer learning generativo (REINVENT4), *Applicability Domain* baseado em embeddings de linguagem química (ChemBERTa) e análise de scaffold consenso (Murcko + MCS) para priorizar candidatos estruturalmente plausíveis a partir de um dataset de 50 ligantes de referência.

## Pipeline

1. **Geração molecular** — REINVENT4 especializado por transfer learning nos 50 ligantes de referência (744 moléculas geradas). *Executado externamente, não incluído neste repositório.*
2. **Filtragem por drug-likeness** — regras de Lipinski + filtro PAINS.
3. **Applicability Domain (ChemBERTa)** — distância no espaço latente de um modelo de linguagem química pré-treinado (`seyonec/ChemBERTa-zinc-base-v1`), via kNN.
4. **Score composto e seleção final** — combinação de confiança generativa (NLL) e proximidade ao espaço de referência, com penalidade por fragmentos estruturais desfavoráveis (BRICS/SAR) e filtro de novidade mínima.
5. **Análise de scaffold consenso** — núcleo farmacofórico dominante via sobreposição de scaffold de Bemis–Murcko e Maximum Common Substructure (MCS).

### Fora deste repositório

- O modelo generativo REINVENT4 em si (etapa de transfer learning)
- O co-folding estrutural com Boltz-2, usado para avaliação final de afinidade dos candidatos
- A ferramenta de scoring de afinidade/seletividade (DTA) — peça central da dissertação de mestrado, mantida privada nesta etapa
- Os compostos candidatos finais (SMILES gerados) — não publicados neste repositório

> **Nota metodológica:** uma abordagem inicial de AD baseada em similaridade de Tanimoto (ECFP4) combinada a um ensemble QSAR foi testada, mas descontinuada por instabilidade entre folds de validação cruzada. A avaliação de afinidade foi delegada ao co-folding estrutural (Boltz-2), e o AD final passou a ser calculado no espaço latente do ChemBERTa.

## Estrutura

```
prot_generative_pipeline.ipynb   # notebook consolidado com o pipeline completo
requirements.txt                 # dependências
scripts_originais_usados/        # scripts .py originais que compõem a versão final
data/                            # não incluído — ver formato abaixo
```

## Dados de entrada

O notebook espera dois CSVs em `data/`, não incluídos neste repositório:

| Arquivo | Conteúdo | Colunas esperadas |
|---|---|---|
| `prot_ligands.csv` | 50 ligantes de referência do PROT | `SMILES`, `pChEMBL` (ou `pActivity`/`pIC50`/`pKi`) |
| `generated_prot.csv` | Saída do REINVENT4 (744 moléculas) | `SMILES`, `SMILES_state`, `NLL` |

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook prot_generative_pipeline.ipynb
```

## Referências principais

- Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction.
- Bemis GW, Murcko MA. The Properties of Known Drugs. 1. Molecular Frameworks.
- Wohlwend J, Cruzeiro G, Corso G, et al. Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction.
