"""
Fallstudie Aufgabe 3 – Trends in der Wissenschaft (DLBDSMLUSL01_D)
Schritt 2: Feature Engineering (TF-IDF), Dimensionsreduktion, Clustering.

Voraussetzungen:
    pip install pandas pyarrow matplotlib scikit-learn joblib
Eingabe:
    daten/arxiv_stichprobe.parquet (aus 01_daten_eda.py)

Voraussetzung: 01_daten_eda.py wurde vorher vollständig ausgeführt.

Aufruf (Iteration und k werden beim Start angegeben, im Code ändert sich nichts):
    python 02_features_clustering.py it1        -> Iteration 1, Wahl von k
    python 02_features_clustering.py it1 30     -> Iteration 1, finales Modell k=30
    python 02_features_clustering.py it2        -> Iteration 2, Wahl von k
    python 02_features_clustering.py it2 30     -> Iteration 2, finales Modell k=30

Iteration 1: Standard-Stoppwörter, URLs bleiben im Text.
Iteration 2: URLs entfernt + Fachstoppwörter (Befund aus Iteration 1:
             ein Cluster aus GitHub-Links, Allerweltswörter in den Top-Begriffen).
"""

import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

print("=" * 70)
print("SCHRITT 2 – Feature Engineering, Dimensionsreduktion, Clustering")
print("=" * 70)

# ---------------------------------------------------------------- Parameter
EINGABE = Path("daten/arxiv_stichprobe.parquet")
AUSGABE = Path("daten")
ABB = Path("abbildungen")
MODELLE = Path("modelle")
for ordner in (AUSGABE, ABB, MODELLE):
    ordner.mkdir(exist_ok=True)

SEED = 42
# --- Iteration und k aus dem Aufruf lesen ---------------------------------
if len(sys.argv) < 2 or sys.argv[1] not in ("it1", "it2"):
    sys.exit(
        "Bitte Iteration angeben, z. B.:  python 02_features_clustering.py "
        "it2   oder   python 02_features_clustering.py it2 30"
    )
LAUF = sys.argv[1]
K_FINAL = int(sys.argv[2]) if len(sys.argv) > 2 else None

FACHSTOPPWOERTER_IT2 = [
    "paper",
    "study",
    "studies",
    "propose",
    "proposed",
    "proposes",
    "approach",
    "approaches",
    "method",
    "methods",
    "based",
    "using",
    "use",
    "used",
    "results",
    "result",
    "show",
    "shows",
    "present",
    "presents",
    "presented",
    "novel",
    "new",
    "existing",
    "work",
    "works",
    "performance",
    "available",
    "let",
    "non",
    "high",
]
if LAUF == "it1":
    URLS_ENTFERNEN = False
    FACHSTOPPWOERTER = []
else:
    URLS_ENTFERNEN = True
    FACHSTOPPWOERTER = FACHSTOPPWOERTER_IT2
# Feature Engineering (TF-IDF)
MIN_DF = 5  # Begriff muss in mind. 5 Dokumenten vorkommen
MAX_DF = 0.5  # Begriffe in > 50 % der Dokumente sind kaum trennscharf
MAX_FEATURES = 30_000  # Obergrenze Vokabular (Feature-Auswahl)
NGRAMME = (1, 2)  # Einzelwörter und Bigramme
# Dimensionsreduktion
SVD_KOMPONENTEN = 100
# Clustering
K_WERTE = list(range(5, 41, 5))  # Kandidaten für die Clusteranzahl
SILHOUETTE_N = 10_000  # Teilstichprobe (Silhouette ist O(n²))
TOP_BEGRIFFE = 12

# ------------------------------------------------------- Daten vorbereiten
df = pd.read_parquet(EINGABE)
print(
    f"Iteration: {LAUF}   k: {K_FINAL or 'Suche'}   URLs entfernen: {URLS_ENTFERNEN}   "
    f"Fachstoppwörter: {len(FACHSTOPPWOERTER)}"
)
print(f"Dokumente: {len(df):,}")

LATEX_MUSTER = r"\$[^$]*\$"  # Formeln wie $\alpha$
URL_MUSTER = (
    r"https?://\S+|www\.\S+"  # Links wie https://github.com/...
    r"|\b[\w.-]+\.(?:com|org|io|net)\b\S*"
)
LATEX = re.compile(LATEX_MUSTER)
URL = re.compile(URL_MUSTER)


def text_bereinigen(titel: str, abstract: str) -> str:
    """Titel + Abstract, Formeln (und ggf. URLs) raus, Kleinschreibung."""
    text = LATEX.sub(" ", f"{titel}. {abstract}")
    if URLS_ENTFERNEN:
        text = URL.sub(" ", text)
    return text.lower()


texte = [text_bereinigen(t, a) for t, a in zip(df["titel"], df["abstract"])]

# ----------------------------------------------- Feature Engineering: TF-IDF
vektorisierer = TfidfVectorizer(
    stop_words=sorted(ENGLISH_STOP_WORDS.union(FACHSTOPPWOERTER)),
    token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",  # nur Wörter ab 3 Buchstaben
    ngram_range=NGRAMME,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,  # 1 + log(tf) dämpft Wiederholungen
)
X_tfidf = vektorisierer.fit_transform(texte)
besetzt = X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])
print(
    f"TF-IDF-Matrix: {X_tfidf.shape[0]:,} x {X_tfidf.shape[1]:,}, "
    f"Anteil Nicht-Null-Werte: {besetzt:.4%}"
)

# ------------------------------------- Dimensionsreduktion: TruncatedSVD (LSA)
# PCA würde die dünn besetzte Matrix zentrieren und damit dicht machen;
# TruncatedSVD arbeitet direkt auf der Sparse-Matrix.
n_komp = min(SVD_KOMPONENTEN, X_tfidf.shape[1] - 1)
lsa = make_pipeline(
    TruncatedSVD(n_components=n_komp, random_state=SEED), Normalizer(copy=False)
)  # Länge 1 -> Euklid ~ Kosinus
X_lsa = lsa.fit_transform(X_tfidf)
svd = lsa.named_steps["truncatedsvd"]
print(
    f"LSA: {n_komp} Komponenten, erklärte Varianz gesamt: "
    f"{svd.explained_variance_ratio_.sum():.1%}"
)


def grafik_speichern(dateiname: str) -> None:
    """Speichert die aktuelle Grafik; Fehler brechen das Skript nicht ab."""
    ziel = ABB / dateiname
    try:
        plt.savefig(ziel, dpi=150)
        print(f"Grafik gespeichert: {ziel}")
    except OSError as fehler:
        print(
            f"WARNUNG: {ziel} konnte nicht gespeichert werden ({fehler}). "
            "Ist die Datei in einem anderen Programm geöffnet?"
        )
    finally:
        plt.close()


rng = np.random.default_rng(SEED)
sil_idx = rng.choice(len(X_lsa), size=min(SILHOUETTE_N, len(X_lsa)), replace=False)


def kmeans(k: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k, n_init=5, batch_size=4096, random_state=SEED
    ).fit(X_lsa)


# ================================================ Durchgang 1: Wahl von k
if K_FINAL is None:
    ergebnisse = []
    for k in K_WERTE:
        if k >= len(sil_idx):
            break
        modell = kmeans(k)
        sil = silhouette_score(X_lsa[sil_idx], modell.labels_[sil_idx])
        ergebnisse.append((k, modell.inertia_, sil))
        print(
            f"k = {k:>3}:  Inertia = {modell.inertia_:>12,.1f}   Silhouette = {sil:.4f}"
        )

    tab = pd.DataFrame(ergebnisse, columns=["k", "inertia", "silhouette"])
    tab.to_csv(AUSGABE / f"wahl_k_{LAUF}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(tab["k"], tab["inertia"], marker="o")
    ax1.set(title="Ellenbogen-Methode", xlabel="k", ylabel="Inertia (WCSS)")
    ax2.plot(tab["k"], tab["silhouette"], marker="o", color="tab:orange")
    ax2.set(title="Silhouetten-Score", xlabel="k", ylabel="Silhouette")
    plt.tight_layout()
    grafik_speichern(f"04_wahl_k_{LAUF}.png")
    print(
        f"\nJetzt k wählen und starten mit:  python 02_features_clustering.py {LAUF} <k>"
    )
    raise SystemExit

# ================================================ Durchgang 2: finales Modell
modell = kmeans(K_FINAL)
df["cluster"] = modell.labels_
sil = silhouette_score(X_lsa[sil_idx], modell.labels_[sil_idx])
print(f"\nk = {K_FINAL}: Silhouette (Teilstichprobe) = {sil:.4f}")

# Top-Begriffe je Cluster: Zentren zurück in den TF-IDF-Raum projizieren
zentren = svd.inverse_transform(modell.cluster_centers_)
begriffe = vektorisierer.get_feature_names_out()
top = {
    c: [begriffe[i] for i in zentren[c].argsort()[::-1][:TOP_BEGRIFFE]]
    for c in range(K_FINAL)
}

uebersicht = (
    df.groupby("cluster")
    .agg(
        anzahl=("id", "size"),
        haeufigste_gruppe=("gruppe", lambda s: s.mode().iat[0]),
        anteil_gruppe=("gruppe", lambda s: s.value_counts(normalize=True).iat[0]),
        haeufigste_kategorie=("hauptkategorie", lambda s: s.mode().iat[0]),
    )
    .assign(top_begriffe=lambda d: d.index.map(lambda c: ", ".join(top[c])))
)
uebersicht = uebersicht.sort_values("anzahl", ascending=False)
pd.set_option("display.max_colwidth", 120, "display.width", 200)
print("\nCluster-Übersicht:")
print(uebersicht.to_string(formatters={"anteil_gruppe": "{:.0%}".format}))
uebersicht.to_csv(AUSGABE / f"cluster_uebersicht_k{K_FINAL}_{LAUF}.csv")

# Externer Abgleich mit den arXiv-Hauptgruppen (fließen NICHT ins Clustering ein)
print(
    f"\nAdjusted Rand Index (vs. Hauptgruppe):  "
    f"{adjusted_rand_score(df['gruppe'], df['cluster']):.3f}"
)
print(
    f"Normalized Mutual Information:          "
    f"{normalized_mutual_info_score(df['gruppe'], df['cluster']):.3f}"
)
pd.Series(
    {
        "iteration": LAUF,
        "k": K_FINAL,
        "vokabular": X_tfidf.shape[1],
        "nicht_null_anteil": besetzt,
        "lsa_varianz": svd.explained_variance_ratio_.sum(),
        "silhouette": sil,
        "ari": adjusted_rand_score(df["gruppe"], df["cluster"]),
        "nmi": normalized_mutual_info_score(df["gruppe"], df["cluster"]),
    }
).to_csv(AUSGABE / f"kennzahlen_k{K_FINAL}_{LAUF}.csv", header=["wert"])
pd.crosstab(df["cluster"], df["gruppe"]).to_csv(
    AUSGABE / f"kreuztabelle_k{K_FINAL}_{LAUF}.csv"
)

# 2D-Visualisierung: PCA auf den LSA-Vektoren (Lektion 3.1)
vis_idx = rng.choice(len(X_lsa), size=min(20_000, len(X_lsa)), replace=False)
pca = PCA(n_components=2, random_state=SEED)
X_2d = pca.fit_transform(X_lsa[vis_idx])
plt.figure(figsize=(9, 7))
# 40 unterscheidbare Farben (tab20 + tab20b), damit sich bei k <= 40 keine
# Farbe wiederholt; vmin/vmax ordnen jedem Cluster genau eine Farbe zu
farben = ListedColormap(
    list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
)
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["cluster"].values[vis_idx],
    cmap=farben,
    vmin=0,
    vmax=39,
    s=3,
    alpha=0.6,
)
for c in range(K_FINAL):
    maske = df["cluster"].values[vis_idx] == c
    if maske.any():
        x, y = np.median(X_2d[maske], axis=0)
        plt.annotate(str(c), (x, y), fontsize=9, weight="bold")
plt.title(
    f"Themenkarte (PCA auf LSA, {pca.explained_variance_ratio_.sum():.1%} der Varianz)"
)
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.tight_layout()
grafik_speichern(f"05_themenkarte_k{K_FINAL}_{LAUF}.png")

# Speichern für Schritt 3 (Trendanalyse)
df.to_parquet(
    AUSGABE / f"stichprobe_mit_clustern_k{K_FINAL}_{LAUF}.parquet", index=False
)
joblib.dump(
    {
        "vektorisierer": vektorisierer,
        "lsa": lsa,
        "kmeans": modell,
        "latex_muster": LATEX_MUSTER,
        "url_muster": URL_MUSTER if URLS_ENTFERNEN else None,
    },
    MODELLE / f"pipeline_k{K_FINAL}_{LAUF}.joblib",
)
print("\nErgebnisse, Grafiken und Modell gespeichert.")
