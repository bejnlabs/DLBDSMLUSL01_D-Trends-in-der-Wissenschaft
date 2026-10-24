"""
Fallstudie Aufgabe 3 – Trends in der Wissenschaft
Schritt 1: arXiv-Metadaten laden, eingrenzen, explorativ untersuchen.

Voraussetzungen:
    pip install pandas pyarrow matplotlib
Eingabe:
    arxiv-metadata-oai-snapshot.json (Kaggle, Cornell-University/arxiv)
Ausgabe:
    daten/arxiv_gefiltert.parquet   – alle Arbeiten im Zeitraum (bereinigt)
    daten/arxiv_stichprobe.parquet  – Zufallsstichprobe für das Clustering
    abbildungen/*.png               – explorative Grafiken
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

print("=" * 70)
print("SCHRITT 1 – Daten laden, bereinigen, explorative Analyse")
print("=" * 70)

# ---------------------------------------------------------------- Parameter
EINGABE = Path("arxiv-metadata-oai-snapshot.json")
START = datetime(2023, 1, 1)        # Beginn des Untersuchungszeitraums
ENDE = None                         # None = alles behalten (auch den
                                    # unvollständigen letzten Monat);
                                    # z. B. datetime(2026, 9, 1) schneidet ab
STICHPROBE_N = 100_000              # Größe der Clustering-Stichprobe
SEED = 42                           # für reproduzierbare Ergebnisse

AUSGABE = Path("daten")
ABB = Path("abbildungen")
AUSGABE.mkdir(exist_ok=True)
ABB.mkdir(exist_ok=True)

# arXiv-Archive, die nicht zur Physik gehören; alles andere zählt als Physik
EIGENE_GRUPPEN = {"cs", "math", "stat", "q-bio", "q-fin", "eess", "econ"}


def hauptgruppe(kategorie: str) -> str:
    """'cs.LG' -> 'cs', 'hep-th' -> 'physics', 'math-ph' -> 'physics'."""
    praefix = kategorie.split(".")[0]
    return praefix if praefix in EIGENE_GRUPPEN else "physics"


def erstes_datum(versionen: list) -> datetime | None:
    """Datum der ersten Version, z. B. 'Mon, 2 Apr 2007 19:18:42 GMT'."""
    try:
        return datetime.strptime(versionen[0]["created"],
                                 "%a, %d %b %Y %H:%M:%S %Z")
    except (IndexError, KeyError, ValueError):
        return None


# ---------------------------------------------------- Einlesen (zeilenweise)
zeilen, gesamt, ohne_datum = [], 0, 0
print(f"Lese {EINGABE} (dauert einige Minuten) ...")
with EINGABE.open(encoding="utf-8") as f:
    for zeile in f:
        gesamt += 1
        if gesamt % 250_000 == 0:              # Lebenszeichen beim Einlesen
            print(f"  ... {gesamt:,} Zeilen gelesen", flush=True)
        p = json.loads(zeile)
        datum = erstes_datum(p.get("versions", []))
        if datum is None:
            ohne_datum += 1
            continue
        if datum < START:
            continue
        kategorien = (p.get("categories") or "").split()
        zeilen.append({
            "id": p["id"],
            "titel": " ".join((p.get("title") or "").split()),
            "abstract": " ".join((p.get("abstract") or "").split()),
            "kategorien": " ".join(kategorien),
            "hauptkategorie": kategorien[0] if kategorien else None,
            "gruppe": hauptgruppe(kategorien[0]) if kategorien else None,
            "anz_kategorien": len(kategorien),
            "datum": datum,
            "anz_versionen": len(p.get("versions", [])),
            "hat_journal_ref": bool(p.get("journal-ref")),
            "hat_doi": bool(p.get("doi")),
        })

df = pd.DataFrame(zeilen)
print(f"Einträge gesamt im Snapshot:     {gesamt:>10,}")
print(f"davon ohne lesbares Datum:       {ohne_datum:>10,}")
print(f"im Zeitraum ab {START:%Y-%m-%d}:      {len(df):>10,}")

# ------------------------------------- letzten (unvollständigen) Monat prüfen
letztes = df["datum"].max()
letzter_monat = pd.Period(letztes, freq="M")
df["monat_vollstaendig"] = df["datum"].dt.to_period("M") < letzter_monat
print(f"letzte Einreichung im Snapshot:  {letztes:%Y-%m-%d}")
print(f"letzter Monat {letzter_monat} ist unvollständig: "
      f"{(~df['monat_vollstaendig']).sum():,} Arbeiten bis Tag {letztes.day}")
if ENDE is not None:
    anz_vorher = len(df)
    df = df[df["datum"] < ENDE]
    print(f"entfernt (ab {ENDE:%Y-%m-%d}):         "
          f"{anz_vorher - len(df):>10,}")

# --------------------------------------------------------------- Bereinigung
vorher = len(df)
abstract_klein = df["abstract"].str.lower()
zurueckgezogen = abstract_klein.str.contains(
    r"this (?:paper|article|submission) has been withdrawn", regex=True)
entfernt_admin = abstract_klein.str.contains("redacted by arxiv", regex=False)
print(f"zurückgezogen (withdrawn):        {zurueckgezogen.sum():>9,}")
print(f"von arXiv entfernt (redacted):    {entfernt_admin.sum():>9,}")
df = df[~(zurueckgezogen | entfernt_admin)]
df = df.drop_duplicates(subset="id")
df["laenge_woerter"] = df["abstract"].str.split().str.len()
print(f"entfernt insgesamt (inkl. Dubletten): {vorher - len(df):>5,}")

# ------------------------------------------------- Deskriptive Statistiken
print("\nAbstract-Länge (Wörter):")
print(df["laenge_woerter"].describe().round(1))
print("\nAnteil mit Journal-Referenz: "
      f"{df['hat_journal_ref'].mean():.1%}  (Hinweis auf Peer-Review)")
print(f"Anteil mit DOI:              {df['hat_doi'].mean():.1%}")

gruppen = df["gruppe"].value_counts()
print("\nArbeiten je Hauptgruppe:")
print(pd.DataFrame({"anzahl": gruppen,
                    "anteil": (gruppen / len(df)).map("{:.1%}".format)}))
print("\nTop 15 Hauptkategorien:")
print(df["hauptkategorie"].value_counts().head(15))

monat = df["datum"].dt.to_period("M")
pro_monat = monat.value_counts().sort_index()
pd.DataFrame({"anzahl": pro_monat,
              "vollstaendig": pro_monat.index < letzter_monat}).to_csv(
    AUSGABE / "arbeiten_pro_monat.csv")
print("\nArbeiten pro Jahr:")
print(df["datum"].dt.year.value_counts().sort_index())
voll = pro_monat[pro_monat.index < letzter_monat]
print(f"\nVollständige Monate: {len(voll)}, Minimum {voll.min():,} "
      f"({voll.idxmin()}), Maximum {voll.max():,} ({voll.idxmax()})")
if letzter_monat in pro_monat.index:
    hochgerechnet = pro_monat[letzter_monat] / letztes.day \
        * letzter_monat.days_in_month
    print(f"Unvollständiger Monat {letzter_monat}: {pro_monat[letzter_monat]:,}"
          f" Arbeiten in {letztes.day} Tagen, hochgerechnet ca. "
          f"{hochgerechnet:,.0f} (nur grobe Schätzung)")

print("\nDie 10 kürzesten Abstracts (zur Sichtprüfung):")
for _, z in df.nsmallest(10, "laenge_woerter").iterrows():
    print(f"  {z['id']:<12} {z['laenge_woerter']:>3} Wörter | "
          f"{z['hauptkategorie']:<12} | {z['abstract'][:90]}")

sehr_kurz = (df["laenge_woerter"] < 10).sum()
print(f"\nAbstracts mit weniger als 10 Wörtern: {sehr_kurz:,} "
      f"({sehr_kurz / len(df):.3%})")

# -------------------------------------------------------------- Speichern
# Daten zuerst sichern, damit ein Fehler bei den Grafiken nicht das
# minutenlange Einlesen wertlos macht.
df.to_parquet(AUSGABE / "arxiv_gefiltert.parquet", index=False)
stichprobe = df.sample(n=min(STICHPROBE_N, len(df)), random_state=SEED)
stichprobe.to_parquet(AUSGABE / "arxiv_stichprobe.parquet", index=False)
print(f"\nGespeichert: {len(df):,} Arbeiten, Stichprobe: {len(stichprobe):,}")

# ------------------------------------------------------------------ Grafiken

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

fig, ax = plt.subplots(figsize=(10, 4))
x = pro_monat.index.to_timestamp()
ax.plot(x[:len(voll)], voll.values, marker=".", label="vollständige Monate")
if letzter_monat in pro_monat.index:
    # letzter Monat gestrichelt und beschriftet, damit der Knick erklärt ist
    # Verbindungslinie ohne Marker, grauer Punkt nur auf dem Teilmonat
    ax.plot(x[len(voll) - 1:], pro_monat.values[len(voll) - 1:],
            linestyle="--", color="tab:gray")
    ax.plot(x[-1:], pro_monat.values[-1:], marker="o", linestyle="none",
            color="tab:gray",
            label=f"{letzter_monat} unvollständig (bis Tag {letztes.day})")
ax.set(title="Neue Arbeiten pro Monat (Datum der ersten Version)",
       xlabel="Monat", ylabel="Anzahl Arbeiten")
ax.legend(loc="upper left")
plt.tight_layout()
grafik_speichern("01_arbeiten_pro_monat.png")

ax = gruppen.sort_values().plot.barh(figsize=(7, 4))
ax.set(title="Arbeiten je Hauptgruppe (erste Kategorie)",
       xlabel="Anzahl Arbeiten", ylabel="Hauptgruppe")
plt.tight_layout()
grafik_speichern("02_hauptgruppen.png")

GRENZE = 400
ueber = (df["laenge_woerter"] > GRENZE).sum()
print(f"Abstracts mit mehr als {GRENZE} Wörtern: {ueber:,} "
      f"(in Abbildung 3 nicht dargestellt)")
ax = df["laenge_woerter"].plot.hist(bins=range(0, GRENZE + 10, 10),
                                    figsize=(7, 4))
ax.set(title="Länge der Abstracts", xlabel="Anzahl Wörter",
       ylabel="Anzahl Arbeiten")
ax.text(0.98, 0.95, f"{ueber:,} Abstracts > {GRENZE} Wörter\nnicht dargestellt",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
plt.tight_layout()
grafik_speichern("03_abstract_laenge.png")
