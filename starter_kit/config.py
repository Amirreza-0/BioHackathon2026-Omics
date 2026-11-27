"""Central configuration for the 'drug action from transcriptomics' task."""
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"
PROC = DATA / "processed"
SEED = 2026

# --- Public data sources ------------------------------------------------------------
# L1000 (978 landmark genes) for ~1,500 compounds in A549 lung cancer cells, 24 h.
# Figshare 13181966 (Way et al. 2022, "L1000 data for LINCS profiling complementarity").
L1000_ZIP_URL = "https://ndownloader.figshare.com/articles/13181966/versions/1"
L1000_LEVEL5 = "level_5_modz_n9482x978.gctx"                  # consensus signature per compound x dose
L1000_LEVEL5_META = "col_meta_level_5_REP.A_A549_only_n9482.txt"

# MoA labels (Drug Repurposing Hub, mapped to Broad IDs), pinned to a fixed commit.
_COMMIT = "061870127481dcd73c29df85ebcfddeac2ed0828"
MOA_URL = (f"https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/{_COMMIT}"
           "/metadata/moa/repurposing_info_external_moa_map_resolved.tsv")

# --- Task definition -------------------------------------------------------------------
# Targeted anti-cancer mechanisms (exact Repurposing Hub MoA strings).
TARGET_CLASSES = [
    "EGFR inhibitor", "HDAC inhibitor", "MEK inhibitor", "CDK inhibitor",
    "mTOR inhibitor", "PI3K inhibitor", "HSP inhibitor", "proteasome inhibitor",
    "RAF inhibitor", "AKT inhibitor", "Aurora kinase inhibitor", "PARP inhibitor",
    "PLK inhibitor", "JAK inhibitor", "tubulin polymerization inhibitor",
    "topoisomerase inhibitor",
]
MIN_COMPOUNDS_PER_CLASS = 5    # classes with fewer compounds (after matching to L1000) are dropped
SINGLE_MOA_ONLY = True         # drop compounds annotated with several mechanisms

# --- Splits ----------------------------------------------------------------------------
TEST_FRAC = 0.25               # fraction of compounds per class held out (hidden test)
MIN_TEST_PER_CLASS = 1
CV_FOLDS = 4                   # grouped CV inside the training set (by compound)
