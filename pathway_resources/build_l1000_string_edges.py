"""Build l1000_string_edges.csv: STRING protein-protein associations among the 978 L1000 landmark genes.

Source: STRING v12.0 (Szklarczyk et al. 2023, Nucleic Acids Res 51:D638, doi:10.1093/nar/gkac1000), human (taxon 9606),
via the version-pinned REST API https://version-12-0.string-db.org/api (endpoints get_string_ids and network).
Only gene symbols are sent. Output columns: probe_a, probe_b, symbol_a, symbol_b, score (STRING combined score, 0-1),
edges with combined score >= 0.4 (STRING "medium confidence"); each undirected edge once.

Run from the project root:  python pathway_resources/build_l1000_string_edges.py
"""
import csv, io, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API, SPECIES, MIN_SCORE = "https://version-12-0.string-db.org/api", 9606, 400

def post(endpoint, **params):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"{API}/tsv/{endpoint}", data=data, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=300) as r: return list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))

rows = list(csv.DictReader(open(HERE / "l1000_pathway_weights.csv")))
probe_of = {}
for r in rows: probe_of.setdefault(r["symbol"], r["probe"])                      # symbol -> first probe
symbols = [s for s in probe_of if s and s != "NA"]

ids = {}                                                                          # STRING id -> our symbol
for i in range(0, len(symbols), 400):
    for r in post("get_string_ids", identifiers="\r".join(symbols[i:i + 400]), species=SPECIES, limit=1, echo_query=1):
        ids[r["stringId"]] = r["queryItem"]
    time.sleep(1)
print(f"{len(symbols)} symbols, {len(ids)} mapped to STRING", file=sys.stderr)

edges = {}
for r in post("network", identifiers="\r".join(ids), species=SPECIES, required_score=MIN_SCORE, network_type="functional"):
    a, b = ids.get(r["stringId_A"]), ids.get(r["stringId_B"])
    if a and b and a != b: edges[tuple(sorted((a, b)))] = max(edges.get(tuple(sorted((a, b))), 0), float(r["score"]))
with open(HERE / "l1000_string_edges.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["probe_a", "probe_b", "symbol_a", "symbol_b", "score"])
    for (a, b), s in sorted(edges.items()): w.writerow([probe_of[a], probe_of[b], a, b, f"{s:.3f}"])
print(f"{len(edges)} edges among {len({g for e in edges for g in e})} genes -> {HERE / 'l1000_string_edges.csv'}", file=sys.stderr)
