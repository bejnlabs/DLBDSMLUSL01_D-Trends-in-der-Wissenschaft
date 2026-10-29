# Trends in der Wissenschaft – Themencluster in arXiv-Arbeiten seit 2023

Fallstudie zum Kurs DLBDSMLUSL01_D (Maschinelles Lernen: Unsupervised Learning und Feature Engineering).

Die Skripte bilden mit TF-IDF, LSA und k-Means Themencluster aus Titeln und Abstracts aller arXiv-Arbeiten
seit Januar 2023 und werten aus, welche Themen ihren Anteil am Gesamtbestand steigern.

## Ergebnisse auf einen Blick

Grundlage sind 1.029.434 Arbeiten (Januar 2023 bis 15.10.2026), gruppiert in 35 Themen (Iteration 2).
Verglichen werden die Anteile im Jahr 2023 mit den letzten zwölf vollständigen Monaten (10/2025–09/2026).

| Thema | Anteil 2023 | Anteil zuletzt | Faktor |
|---|---|---|---|
| KI-Agenten | 0,74 % | 4,10 % | 5,53 |
| Retrieval-Augmented Generation | 0,76 % | 2,60 % | 3,43 |
| Robotik, Manipulation | 1,24 % | 2,90 % | 2,34 |
| Große Sprachmodelle | 3,02 % | 6,47 % | 2,15 |
| Bildgenerierung, Diffusion | 2,72 % | 4,00 % | 1,47 |
| Neuronale Netze, Graph Learning | 5,16 % | 3,12 % | 0,61 |

- Den stärksten Zuwachs zeigen KI-Agenten. Der Befund ist in beiden Iterationen nahezu gleich (Faktor 5,4 bzw. 5,5).
- Der Anteil der großen Sprachmodelle erreichte Mitte 2025 knapp 9 % und ist seitdem gesunken.
- Ein sinkender Anteil heißt nicht, dass ein Thema absolut schrumpft: arXiv wuchs im Zeitraum um gut 16 % pro Jahr.

Die Namen der Themen sind eine eigene Interpretation der wichtigsten Begriffe je Cluster, keine offizielle Einteilung.

## Methode in Kürze

1. **Daten:** Metadaten ab 01.01.2023, Datum der ersten Version, Bereinigung um zurückgezogene und von arXiv entfernte Arbeiten.
2. **Merkmale:** TF-IDF auf Titel und Abstract (Einzelwörter und Bigramme, max. 30.000 Begriffe).
3. **Dimensionsreduktion:** TruncatedSVD (LSA) auf 100 Komponenten, Normierung auf Länge 1; PCA für die 2D-Themenkarte.
4. **Clustering:** Mini-Batch-k-Means, Wahl von k über Ellenbogen-Methode und Silhouetten-Score, trainiert auf einer Stichprobe von 100.000 Arbeiten.
5. **Trends:** Zuordnung aller Arbeiten zu den Clustern, Vergleich der monatlichen Anteile.
6. **Iterationen:** Iteration 1 (k = 30) ergab einen Cluster aus GitHub-Links. Iteration 2 (k = 35) entfernt URLs und 31 Fachstoppwörter.

## Daten

Die Metadaten stammen aus dem Kaggle-Datensatz von Cornell University:
https://www.kaggle.com/datasets/Cornell-University/arxiv

Die Datei `arxiv-metadata-oai-snapshot.json` (rund 1,3 GB) ist nicht im Repository enthalten und muss
vor dem ersten Lauf in den Projektordner gelegt werden. Der Datensatz wird wöchentlich aktualisiert;
Der Snapshot wurde mit Einreichungen bis zum 15.10.2026 ausgewertet. Mit einem neueren Snapshot
weichen die Zahlen entsprechend ab.

## Ablauf

```
python 01_daten_eda.py                      # Einlesen, Bereinigung, explorative Analyse
python 02_features_clustering.py it1        # Iteration 1: Suche nach k
python 02_features_clustering.py it1 30     # Iteration 1: finales Modell
python 02_features_clustering.py it2        # Iteration 2: Suche nach k
python 02_features_clustering.py it2 35     # Iteration 2: finales Modell
python 02_features_clustering.py it2 30     # Kontrollmodell mit gleichem k wie Iteration 1
python 03_trends.py it2 35                  # Trendanalyse (Hauptmodell)
python 03_trends.py it1 30                  # Trendanalyse zur Robustheitsprüfung
python 04_vergleich_iterationen.py 30 30    # Vergleich der Iterationen
python 05_cluster_pruefen.py it2 35         # Titel und Kategorien je Cluster zur Benennung
```

Schritt 1 und die Zuordnung in Schritt 3 dauern je einige Minuten, die übrigen Schritte deutlich kürzer.

## Reproduzierbarkeit

Stichprobe, k-Means und Teilstichproben verwenden den festen Startwert 42. Zwei unabhängige Läufe auf
demselben Snapshot lieferten identische Ergebnisse.

## Ordner

| Ordner | Inhalt | Im Repository |
|---|---|---|
| `daten/` | Ergebnistabellen (CSV) | ja |
| `daten/*.parquet` | bereinigte Daten und Stichproben | nein, werden lokal erzeugt |
| `abbildungen/` | alle Grafiken der Auswertung | ja |
| `modelle/` | trainierte Modelle | nein, werden lokal erzeugt |

## Grenzen

- TF-IDF erfasst Wortformen, nicht Bedeutung. Ein sinkender Anteil kann auch auf veränderte Begriffe zurückgehen.
- Jede Arbeit gehört genau einem Cluster an, obwohl viele Arbeiten mehrere Themen berühren.
- Die Silhouetten-Werte liegen unter 0,08, die Themen überlappen also stark.
- arXiv bildet vor allem Informatik, Physik und Mathematik ab; nur 14 % der Arbeiten tragen eine Journal-Referenz.