# Pathway resources

`l1000_pathway_weights.csv` — 978 L1000 landmark probes × 52 pathway readouts, used by
`src/model_benchmark.ipynb` for the biology check. Rows: `probe` (HG-U133A probe ID, same
order as the columns of `X_train.csv.gz`), `symbol`. Columns:

| Prefix | n | Content | Source |
|---|---|---|---|
| `PROGENy:` | 13 | PROGENy footprint weights, top 500 genes per pathway by p-value (Trail dropped: 4 L1000 genes) | Schubert et al. 2018, *Nat Commun* 9:20, doi:10.1038/s41467-017-02391-6 — Bioconductor `progeny` 1.32.0, object `model_human_full` (https://bioconductor.org/packages/progeny) |
| `H:` | 38 | MSigDB Hallmark sets with ≥10 L1000 genes, weight 1/n per member gene | Liberzon et al. 2015, *Cell Syst* 1:417–425, doi:10.1016/j.cels.2015.12.004 — MSigDB release 2026.1 via R `msigdbr` 26.1.1 (https://www.gsea-msigdb.org/gsea/msigdb) |
| `custom:` | 1 | `HSF1_heat_shock`: HSPA1A, DNAJB1, HSPA8, HSPD1, HSPA4, BAG3, HSPB1 (weight 1/7) | Hand-picked canonical HSF1 targets present on L1000; not from a database |

Probe → symbol: Bioconductor `hgu133a.db` 3.13.0 (https://bioconductor.org/packages/hgu133a.db);
10 probes map to >1 symbol, the first is kept.

## How the data are obtained (nothing is downloaded by hand)
- **PROGENy** and **hgu133a.db**: the data ship inside the conda/Bioconductor packages.
- **MSigDB Hallmark**: on first call `msigdbr` 26.1.1 downloads the MSigDB 2026.1 archive from Zenodo,
  https://zenodo.org/records/18968178/files/msigdb.2026.1.zip (md5 `512ba99c6827141a9d471972b812d4ac`,
  checked by the package), and caches it in `tools::R_user_dir("msigdbr", "cache")`.

## Rebuild
```bash
conda create -n pathways-r -c conda-forge -c bioconda r-base=4.5 \
    bioconductor-progeny=1.32.0 bioconductor-hgu133a.db=3.13.0 r-msigdbr=26.1.1 r-dplyr r-readr
conda activate pathways-r
Rscript pathway_resources/build_l1000_pathway_weights.R              # from the project root
shasum -a 256 pathway_resources/l1000_pathway_weights.csv
# expected: 0f6f6c997ed6800ca39501a25105462eba60aa69656bbd3fd62489b3e122f30f
```
Built with R 4.5.3, AnnotationDbi 1.72.0, dplyr 1.2.1; the rebuild was verified byte-identical.

## How the readouts are used (in the notebook)
Signature score = z · w, standardized against 500 gene-label permutations; compound profile = mean over
doses ≥ 3.33 µM, robust-z over training compounds. A call is *pathway-supported* when the expected
readout(s) of the called class (the `PRIORS` dict in the notebook — the author's mapping from target
biology, not from these databases) reach robust z ≥ 1. HDAC has no readout.
