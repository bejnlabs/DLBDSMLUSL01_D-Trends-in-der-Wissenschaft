"""
Fallstudie Aufgabe 3 – Trends in der Wissenschaft
Schritt 3: Trendanalyse der Themencluster über die Zeit.

Voraussetzung: 01_daten_eda.py und 02_features_clustering.py (finales Modell)
wurden vorher vollständig ausgeführt.

Aufruf:  python 03_trends.py it2 30     (Iteration und k wie in Schritt 2)

Voraussetzungen:
    pip install pandas pyarrow matplotlib scikit-learn joblib
Eingabe:
    daten/arxiv_gefiltert.parquet           (aus Schritt 1)
    modelle/pipeline_k<K>.joblib            (aus Schritt 2)
    daten/cluster_uebersicht_k<K>.csv       (aus Schritt 2)
Ausgabe:
    (alle Dateinamen zusätzlich mit Iterations-Kennung, z. B. _it2)
    daten/trends_k<K>.csv                   – Wachstum je Cluster
    daten/anteile_pro_monat_k<K>.csv        – Monatsanteile je Cluster
    abbildungen/06_trends_top_k<K>.png      – stärkste Gewinner/Verlierer
    abbildungen/07_veraenderung_k<K>.png    – Veränderung aller Cluster
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 70)
print("SCHRITT 3 – Trendanalyse der Themencluster")
print("=" * 70)

# ---------------------------------------------------------------- Parameter
if len(sys.argv) < 3 or sys.argv[1] not in ("it1", "it2"):
    sys.exit("Bitte Iteration und k angeben, z. B.:  python 03_trends.py it2 30")
LAUF = sys.argv[1]           # muss zum Modell aus Schritt 2 passen
K_FINAL = int(sys.argv[2])
BLOCKGROESSE = 50_000        # Dokumente pro Rechenblock (schont den Speicher)
VERGLEICH_MONATE = 12        # erste vs. letzte 12 vollständige Monate
TOP_N = 5                    # Anzahl Gewinner/Verlierer in der Grafik

DATEN = Path("daten")
ABB = Path("abbildungen")
ABB.mkdir(exist_ok=True)


def grafik_speichern(dateiname: str) -> None:
    """Speichert die aktuelle Grafik; Fehler brechen das Skript nicht ab."""
    ziel = ABB / dateiname
    try:
        plt.savefig(ziel, dpi=150)
        print(f"Grafik gespeichert: {ziel}")
    except OSError as fehler:
        print(f"WARNUNG: {ziel} konnte nicht gespeichert werden ({fehler}). "
              "Ist die Datei in einem anderen Programm geöffnet?")
    finally:
        plt.close()


# ------------------------------------------------------------- Laden
df = pd.read_parquet(DATEN / "arxiv_gefiltert.parquet",
                     columns=["id", "titel", "abstract", "datum", "gruppe",
                              "monat_vollstaendig"])
modelldatei = Path("modelle") / f"pipeline_k{K_FINAL}_{LAUF}.joblib"
if not modelldatei.exists():
    sys.exit(f"Modell {modelldatei} fehlt. Zuerst ausführen:  "
             f"python 02_features_clustering.py {LAUF} {K_FINAL}")
modell = joblib.load(modelldatei)
uebersicht = pd.read_csv(DATEN / f"cluster_uebersicht_k{K_FINAL}_{LAUF}.csv",
                         index_col="cluster")
print(f"Arbeiten: {len(df):,}   Modell: k = {K_FINAL}, Iteration {LAUF}")

# Kurzbezeichnung je Cluster aus den drei wichtigsten Begriffen
bezeichnung = {c: f"C{c}: " + ", ".join(t.split(", ")[:3])
               for c, t in uebersicht["top_begriffe"].items()}

# ------------------------- alle Arbeiten den Clustern zuordnen (blockweise)
# Gleiche Textbereinigung wie in Schritt 2 (Regeln stammen aus dem Modell),
# sonst "sieht" das Modell die Texte anders als beim Training.
texte = (df["titel"] + ". " + df["abstract"]).str.replace(
    modell["latex_muster"], " ", regex=True)
if modell.get("url_muster"):
    texte = texte.str.replace(modell["url_muster"], " ", regex=True)
texte = texte.str.lower()

cluster = np.empty(len(df), dtype=np.int32)
for start in range(0, len(df), BLOCKGROESSE):
    block = texte.iloc[start:start + BLOCKGROESSE]
    X = modell["lsa"].transform(modell["vektorisierer"].transform(block))
    cluster[start:start + BLOCKGROESSE] = modell["kmeans"].predict(X)
    print(f"  ... {min(start + BLOCKGROESSE, len(df)):,} zugeordnet",
          flush=True)
df["cluster"] = cluster

# --------------------------------- Monatsanteile (nur vollständige Monate)
# Anteile statt absoluter Zahlen: arXiv wächst insgesamt, ein Thema ist nur
# dann ein Trend, wenn sein ANTEIL steigt.
voll = df[df["monat_vollstaendig"]].copy()
print(f"\nausgeschlossen (unvollständiger Monat): "
      f"{(~df['monat_vollstaendig']).sum():,}")
voll["monat"] = voll["datum"].dt.to_period("M")
anzahl = pd.crosstab(voll["monat"], voll["cluster"])
anteile = anzahl.div(anzahl.sum(axis=1), axis=0)
anteile.to_csv(DATEN / f"anteile_pro_monat_k{K_FINAL}_{LAUF}.csv")

# ---------------------------------------------- Wachstum je Cluster messen
anfang = anteile.iloc[:VERGLEICH_MONATE].mean()
ende = anteile.iloc[-VERGLEICH_MONATE:].mean()
# lineare Steigung des Monatsanteils, umgerechnet auf Prozentpunkte pro Jahr
t = np.arange(len(anteile))
steigung = pd.Series({c: np.polyfit(t, anteile[c].values, 1)[0] * 12 * 100
                      for c in anteile.columns})

trends = pd.DataFrame({
    "bezeichnung": pd.Series(bezeichnung),
    "anteil_anfang_%": anfang * 100,
    "anteil_ende_%": ende * 100,
    "veraenderung_pp": (ende - anfang) * 100,
    "faktor": ende / anfang,
    "steigung_pp_pro_jahr": steigung,
    "haeufigste_gruppe": uebersicht["haeufigste_gruppe"],
}).sort_values("veraenderung_pp", ascending=False)
trends.index.name = "cluster"
trends.to_csv(DATEN / f"trends_k{K_FINAL}_{LAUF}.csv")

pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(f"\nVergleich: erste {VERGLEICH_MONATE} Monate "
      f"({anteile.index[0]}–{anteile.index[VERGLEICH_MONATE - 1]}) vs. "
      f"letzte {VERGLEICH_MONATE} Monate "
      f"({anteile.index[-VERGLEICH_MONATE]}–{anteile.index[-1]})")
print(trends.round({"anteil_anfang_%": 2, "anteil_ende_%": 2,
                    "veraenderung_pp": 2, "faktor": 2,
                    "steigung_pp_pro_jahr": 2}).to_string())

# Plausibilitätsprüfung: Anteile summieren sich je Monat zu 100 %
abweichung = (anteile.sum(axis=1) - 1).abs().max()
print(f"\nKontrolle: max. Abweichung der Monatssumme von 100 %: "
      f"{abweichung:.2e}")

# ------------------- Sonderfall: welche Cluster tragen den stärksten Monat?
gesamt = anzahl.sum(axis=1)
spitze = gesamt.iloc[6:].idxmax()      # erst ab Monat 7: 6 Vormonate nötig
pos = gesamt.index.get_loc(spitze)
vorher = anzahl.iloc[pos - 6:pos].mean()          # Schnitt der 6 Vormonate
zuwachs = (anzahl.loc[spitze] - vorher).sort_values(ascending=False)
print(f"\nStärkster Monat: {spitze} mit {gesamt[spitze]:,} Arbeiten "
      f"(Schnitt der 6 Vormonate: {gesamt.iloc[pos - 6:pos].mean():,.0f})")
print("Größte Zuwächse gegenüber den Vormonaten:")
for c, wert in zuwachs.head(5).items():
    print(f"  {bezeichnung[c]:<55} {wert:+,.0f}")

# ------------------------------------------------------------- Grafiken
gewinner = trends.index[:TOP_N]
verlierer = trends.index[-TOP_N:]
geglaettet = anteile.rolling(3, center=True, min_periods=1).mean() * 100
x = anteile.index.to_timestamp()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for c in gewinner:
    ax1.plot(x, geglaettet[c], label=bezeichnung[c])
for c in verlierer:
    ax2.plot(x, geglaettet[c], label=bezeichnung[c])
ax1.set(title=f"Stärkster Zuwachs (Top {TOP_N})", xlabel="Monat",
        ylabel="Anteil an allen Arbeiten in %")
ax2.set(title=f"Stärkster Rückgang (Top {TOP_N})", xlabel="Monat",
        ylabel="Anteil an allen Arbeiten in %")
ax1.legend(fontsize=7)
ax2.legend(fontsize=7)
fig.suptitle("Monatlicher Anteil je Themencluster (gleitender 3-Monats-Mittelwert)")
plt.tight_layout()
grafik_speichern(f"06_trends_top_k{K_FINAL}_{LAUF}.png")

sortiert = trends.sort_values("veraenderung_pp")
farben = ["tab:green" if v > 0 else "tab:red"
          for v in sortiert["veraenderung_pp"]]
fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(sortiert))))
ax.barh(sortiert["bezeichnung"], sortiert["veraenderung_pp"], color=farben)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(title=f"Veränderung des Anteils: erste vs. letzte "
             f"{VERGLEICH_MONATE} Monate",
       xlabel="Veränderung in Prozentpunkten", ylabel="")
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
grafik_speichern(f"07_veraenderung_k{K_FINAL}_{LAUF}.png")

print("\nSchritt 3 abgeschlossen.")
