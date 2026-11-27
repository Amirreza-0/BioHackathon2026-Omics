# Rebuild pathway_resources/l1000_pathway_weights.csv from public sources.
#   Rscript pathway_resources/build_l1000_pathway_weights.R          (run from the project root)
# Environment (conda): see README.md in this folder.
suppressMessages({library(hgu133a.db); library(progeny); library(msigdbr); library(dplyr)})
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "."
out  <- if (length(args) >= 2) args[2] else file.path(root, "pathway_resources", "l1000_pathway_weights.csv")

# 1. L1000 landmark probes (HG-U133A probe IDs) in the column order of the challenge data
probes <- strsplit(readLines(gzfile(file.path(root, "X_test.csv.gz")), n = 1), ",")[[1]][-1]
map <- AnnotationDbi::select(hgu133a.db, keys = probes, columns = "SYMBOL", keytype = "PROBEID")
map <- map[!duplicated(map$PROBEID), ]                     # 10 probes map to >1 symbol: keep the first

# 2. PROGENy footprints: top 500 genes per pathway by p-value (package default); Trail dropped (4 L1000 genes)
data("model_human_full", package = "progeny")
pg <- model_human_full %>% group_by(pathway) %>% slice_min(p.value, n = 500, with_ties = FALSE) %>%
  ungroup() %>% filter(pathway != "Trail")

# 3. MSigDB Hallmark (msigdbr downloads the MSigDB release from Zenodo on first use, md5-checked)
hm <- msigdbr(species = "Homo sapiens", collection = "H")

# 4. Weight matrix: PROGENy weights; Hallmark and HSF1 sets as 1/n over member genes on L1000
W <- list()
for (p in sort(unique(as.character(pg$pathway)), method = "radix")) { s <- pg[pg$pathway == p, ]; w <- s$weight[match(map$SYMBOL, s$gene)]; w[is.na(w)] <- 0; W[[paste0("PROGENy:", p)]] <- w }
for (h in sort(unique(as.character(hm$gs_name)), method = "radix")) { inn <- map$SYMBOL %in% unique(hm$gene_symbol[hm$gs_name == h]); if (sum(inn) >= 10) W[[paste0("H:", sub("HALLMARK_", "", h))]] <- inn / sum(inn) }
hs <- map$SYMBOL %in% c("HSPA1A", "DNAJB1", "HSPA8", "HSPD1", "HSPA4", "BAG3", "HSPB1")   # custom HSF1 heat-shock targets
W[["custom:HSF1_heat_shock"]] <- hs / sum(hs)

write.csv(data.frame(probe = map$PROBEID, symbol = map$SYMBOL, as.data.frame(W, check.names = FALSE), check.names = FALSE), out, row.names = FALSE)
cat("wrote", out, ":", nrow(map), "probes x", length(W), "readouts\n")
for (p in c("progeny", "hgu133a.db", "msigdbr", "AnnotationDbi", "dplyr")) cat(p, as.character(packageVersion(p)), "\n")
