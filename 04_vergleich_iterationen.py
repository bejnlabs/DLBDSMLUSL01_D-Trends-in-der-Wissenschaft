"""
Fallstudie Aufgabe 3 – Trends in der Wissenschaft
Schritt 4: Vergleich von Iteration 1 und Iteration 2.

Voraussetzung: 02_features_clustering.py wurde für beide Iterationen mit
finalem k ausgeführt (z. B. "it1 30" und "it2 30").

Aufruf:  python 04_vergleich_iterationen.py <k_it1> <k_it2>
         z. B. python 04_vergleich_iterationen.py 30 30

Ausgabe:
    daten/vergleich_kennzahlen.csv     – Kennzahlen beider Iterationen
    daten/vergleich_zuordnung.csv      – welcher it2-Cluster entspricht welchem it1-Cluster
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

print("=" * 70)
print("SCHRITT 4 – Vergleich Iteration 1 / Iteration 2")
print("=" * 70)

if len(sys.argv) < 3:
    sys.exit("Bitte k beider Iterationen angeben, z. B.:  "
             "python 04_vergleich_iterationen.py 30 30")
K1, K2 = int(sys.argv[1]), int(sys.argv[2])
DATEN = Path("daten")

# Allerweltswörter und URL-Reste, deren Auftreten in den Top-Begriffen
# gezählt wird (Liste = Fachstoppwörter aus Iteration 2 + URL-Bestandteile)
URL_TEILE = {"github", "https", "http", "com", "www", "github com",
             "https github", "available https"}
ALLERWELT = {"paper", "study", "studies", "propose", "proposed", "proposes",
             "approach", "approaches", "method", "methods", "based", "using",
             "use", "used", "results", "result", "show", "shows", "present",
             "presents", "presented", "novel", "new", "existing", "work",
             "works", "performance", "available", "let", "non", "high"}


def laden(lauf: str, k: int):
    datei = DATEN / f"kennzahlen_k{k}_{lauf}.csv"
    if not datei.exists():
        sys.exit(f"{datei} fehlt. Zuerst ausführen:  "
                 f"python 02_features_clustering.py {lauf} {k}")
    kenn = pd.read_csv(DATEN / f"kennzahlen_k{k}_{lauf}.csv", index_col=0)["wert"]
    ueb = pd.read_csv(DATEN / f"cluster_uebersicht_k{k}_{lauf}.csv",
                      index_col="cluster")
    cl = pd.read_parquet(DATEN / f"stichprobe_mit_clustern_k{k}_{lauf}.parquet",
                         columns=["id", "cluster"])
    return kenn, ueb, cl


def zaehlen(ueb: pd.DataFrame, woerter: set) -> int:
    """Wie oft kommen diese Wörter in allen Top-Begriff-Listen vor?"""
    return sum(t in woerter for liste in ueb["top_begriffe"]
               for t in liste.split(", "))


kenn1, ueb1, cl1 = laden("it1", K1)
kenn2, ueb2, cl2 = laden("it2", K2)

# ------------------------------------------------------- Kennzahlen
vergleich = pd.DataFrame({"Iteration 1": kenn1, "Iteration 2": kenn2})
vergleich.loc["url_begriffe_in_top"] = [zaehlen(ueb1, URL_TEILE),
                                        zaehlen(ueb2, URL_TEILE)]
vergleich.loc["allerweltswoerter_in_top"] = [zaehlen(ueb1, ALLERWELT),
                                             zaehlen(ueb2, ALLERWELT)]
vergleich.loc["cluster_mit_90%_einer_gruppe"] = [
    (ueb1["anteil_gruppe"] >= 0.9).sum(), (ueb2["anteil_gruppe"] >= 0.9).sum()]
vergleich.to_csv(DATEN / "vergleich_kennzahlen.csv")


def runden(wert):
    try:
        return f"{float(wert):.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return wert


print("\nKennzahlen:")
print(vergleich.map(runden).to_string())

# ------------------------------------- Übereinstimmung der beiden Einteilungen
beide = cl1.merge(cl2, on="id", suffixes=("_it1", "_it2"))
print(f"\nGemeinsame Dokumente: {len(beide):,}")
print(f"Übereinstimmung it1 vs. it2:  ARI = "
      f"{adjusted_rand_score(beide['cluster_it1'], beide['cluster_it2']):.3f}"
      f"   NMI = "
      f"{normalized_mutual_info_score(beide['cluster_it1'], beide['cluster_it2']):.3f}")

# Für jeden it2-Cluster: aus welchem it1-Cluster stammen die meisten Dokumente?
kreuz = pd.crosstab(beide["cluster_it2"], beide["cluster_it1"])
zuordnung = pd.DataFrame({
    "top_begriffe_it2": ueb2["top_begriffe"].str.split(", ").str[:5]
                                            .str.join(", "),
    "groesster_it1_cluster": kreuz.idxmax(axis=1),
    "anteil": kreuz.max(axis=1) / kreuz.sum(axis=1),
})
zuordnung["top_begriffe_it1"] = zuordnung["groesster_it1_cluster"].map(
    ueb1["top_begriffe"].str.split(", ").str[:5].str.join(", "))
zuordnung = zuordnung.sort_values("anteil")
zuordnung.to_csv(DATEN / "vergleich_zuordnung.csv")
pd.set_option("display.width", 220, "display.max_colwidth", 55)
print("\nZuordnung it2 -> it1 (oben: stärkste Veränderung):")
print(zuordnung.to_string(formatters={"anteil": "{:.0%}".format}))

# ------------------------------- Wohin sind die Dokumente des URL-Clusters?
url_cluster = [c for c, t in ueb1["top_begriffe"].items()
               if any(w in URL_TEILE for w in t.split(", ")[:5])]
for c in url_cluster:
    teil = beide[beide["cluster_it1"] == c]["cluster_it2"].value_counts(
        normalize=True).head(5)
    print(f"\nit1-Cluster {c} ({ueb1.loc[c, 'top_begriffe'][:60]} ...)"
          f" verteilt sich in it2 auf:")
    for c2, a in teil.items():
        print(f"  it2-Cluster {c2:>2} ({a:.0%}): "
              f"{', '.join(ueb2.loc[c2, 'top_begriffe'].split(', ')[:5])}")

print("\nSchritt 4 abgeschlossen.")
