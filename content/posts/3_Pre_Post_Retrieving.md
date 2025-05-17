+++
 date = '2025-05-03T08:26:23+02:00'
 draft = false
 math = true
 number-sections = true
 disableComments = false
 title = 'Pre- & Post-Retrieving'
+++

Retrieval-Augmented Generation (RAG) kombiniert die Stärken großer Sprachmodelle (LLMs) mit externen Wissensquellen, um präzisere und aktuellere Antworten zu liefern. Während klassische LLMs ausschließlich auf ihren internen Parametern basieren, ermöglicht RAG das dynamische Einbinden dokumentierter Informationen - von News-Artikeln bis hin zu Fachpublikationen. Ein typischer RAG-Workflow besteht aus drei Schritten:

- Indexierung: Aufbereitung und Vektorisierung von Dokumenten.
- Retrieval: Auswahl relevanter Dokumentenschnipsel (Chunks) mittels Vektor-Ähnlichkeitssuche.
- Generierung: Einspeisen von Query und Chunks in das LLM für die finale Antwortformulierung.

Doch gerade im Retrieval-Schritt verstecken sich knifflige Herausforderungen: Ungenaue Suchergebnisse, redundante oder irrelevante Informationen und eine Überfrachtung der Kontextlänge können die Qualität der Generierung beeinträchtigen. In der Forschungsübersicht von [Gao et al. (2024)](https://arxiv.org/pdf/2312.10997) werden daher **Pre-Retrieval-** und **Post-Retrieval-Optimierungen** als Kernkomponenten des sogenannten *Advanced RAG* identifiziert, um diese Schwachstellen zu adressieren


## Pre-Retrieval
Vor dem eigentlichen Abruf (*Retrieval*) geht es darum, sowohl die Dokumentenbasis als auch die Nutzeranfrage selbst so aufzubereiten, dass die anschließende Suche möglichst zielgenau verläuft.

### Index Optimierung

<!-- Feingranulare Chunk-Segmentation -->
Bei der Aufteilung der Dokumente (*Chunk Splitting*) werden die Dokumente nicht in willkürlicher Länger, sondern in homogenen Abschnitten aufgeteilt. Das erhöht die Wahrscheinlichkeit, dass die Vektor-Darstellungen tatsächlich thematische Treff­sicherheit bieten. 

Darüber hinaus können noch Metadaten extrahiert werden.
Zusätzliche Informationen (z. B. Dokumenttyp, Veröffentlichungsdatum, Autor) können bei der späteren Filterung helfen, irrelevante Schnipsel auszuschließen.

### Query Transformation
Eine gute Nutzeranfrage ist ein Erfolgstreiber für RAG. Denn wenn die Anfrage schon schlecht gestellt ist, dann ist das Zurückgeben von relevanten Dokumentenabschnitte ebenfalls schlecht. Daher kann es vorteilhaft sein die Nutzeranfrage nochmals anzupassen, um sie für das Retrieving zu optimieren.

Beim *Query Rewriting* wird die ursprüngliche Nutzerfrage mittels NLP-Techniken und LLMs umgeformt, um Mehrdeutigkeiten zu reduzieren und Suchbegriffe fokussierter zu gestalten.

*Query Expansion* beschreibt das Hinzufügen von semantisch verwandter Begriffe (Thesaurus, Wortnetze), um relevante Dokumente zu finden, die ohne Expansion verborgen bleiben.

Außerdem kann die Query angepasst werden. Synonym-Ersetzung oder Umformulierung in prägnantere Suchanfragen verbessern das Retrieving.


## Post-Retrieval
Nachdem eine erste Auswahl an Dokumentenabschnitte vorgenommen wurde, muss die Fülle an Informationen aufbereitet werden und für den Prompt für das LLM optimal vorbereitet werden. Das ist *Post-Retrieval*.

### Context Compression
Mittels Textzusammenfassung oder Extraktionsalgorithmen wird der Dokumentkontext auf das Wesentliche reduziert. So verhindert man, dass irrelevante Details das LLM ablenken.

Außerdem werden Ähnliche oder doppelte Informationen entfernt, um die Prompt-Länge sinnvoll auszunutzen und Overload zu vermeiden.

Je nach Anforderungen passt sich die Menge der Chunks dynamisch an: Fragen, die nur kurze Fakten benötigen, bekommen weniger, dafür aber sehr fokussierte Passagen.

### Zusammenführung der Retriever-Methoden
Wenn innerhalb der RAG-Pipeline mehrere Retriever genutzt, muss das Ergebnis zuletzt zusammengeführt werden, um ein finales Ranking der Dokumentenabschnitte nach Relevanz durchzuführen. 

Dazu können die Scores der Retriever je Dokumentenabschnitt summiert werden. Dabei muss berücksichtigt werden, dass die Scores der Retriever normalisiert sind. Manche Retriever-Methoden wie die Cosine Similarity sind auf den Zahlenbereich $[-1, 1]$ beschränkt. Währenddessen liegt der Score von der Keyword-Suche mit BM25 zwischen $[0, \infty]$. 

Eine andere Methode ist Reciprocal Rank Fusion.

#### Reciprocal Rank Fusion (RFF)
[Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) ist eine einfache Methode zur Kombination mehrerer unterschiedlicher Rangordnungsmethoden im Bereich des Information Retrieval. RRF sortiert die Dokumente einfach nach dem Rang der einzelnen Dokumente in allen Retriever-Sets:
$$
score (d\in D) = \sum_{r \in R}{} \frac{1}{k+r(d)}
$$
mit $k = 60$. Ein größeres $k$ relativiert die Rank-Distanz. 
RRF lässt den räumliche Abstand zwischen den Punktzahlen des ersten und zweiten Ranges bei RRF unberücksichtigt. Bei der Ähnlichkeitssuche (linkes Diagramm in der folgenden Grafik) hat beispielsweise der erste Rang einen Wert von 0,9 und der zweite Rang einen Wert von 0,3. Wenn wir $k$ vernachlässigen und den RRF-Score (rechtes Diagramm in Folgegrafik) für das erste und zweite Dokument berechnen, sehen wir, dass der Unterschied zwischen $\frac{1}{1}= 1$ für den ersten Rang und $\frac{1}{2} = 0.5$ für den zweiten Rang kleiner ist als der Unterschied der Cosine Similarity zwischen dem ersten und zweiten Rang, wie in der folgenden Grafik dargestellt:
<p style="text-align: center;">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/3ce21449-4252-4064-a7a8-2aa38874d55f", width="80%" title = "Left: Score by method (like cosine-similarity) of datapoints - Rigth: The RRF-score for each datapoint/value">
</p>
Das bedeutet, dass wir durch RRF Informationen verlieren. 

### Lost-in-the-Middle
Sprachmodelle haben Schwierigkeiten lange Kontexte zu verarbeiten, dass hat [Nelson F. Liu et. al.](https://arxiv.org/abs/2307.03172) herausgefunden. Daraus wird deutlich, dass Modelle besser darin sind, relevante Informationen zu nutzen, die ganz am Anfang oder am Ende des Eingabekontextes vorkommen und das die Leistung deutlich abnimmt, wenn Modelle auf Informationen in der Mitte des Eingabekontextes zugreifen und diese nutzen müssen. Das bedeutet für das Re-Ranking, dass die abgerufenen Dokumente in einer bestimmten Reihenfolge gemäß dem folgenden Diagramm angeordnet sein müssen:

<p style="text-align: center;">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/c2c2cd86-5b8a-44ea-9fd4-6b44bd3a0891", width="40%">
</p>

Diese Grafik zeigt, dass das vierte relevante Dokument an letzter Stelle stehen muss. Auf der Grundlage der Arbeit von [Nelson F. Liu et. al.](https://arxiv.org/abs/2307.03172) implementiere ich einen Long-Context-Reranker, da ich aus eigener Erfahrung damit bessere Ergebnisse erzielt habe.

## Zusammenfassung
Retrieval-Augmented Generation bietet enorme Potenziale, um die Grenzen klassischer Sprachmodelle zu überwinden – vorausgesetzt, das zugrunde liegende Retrieval ist präzise, effizient und robust. Dabei zeigt sich: Die Qualität der Antwort steht und fällt nicht nur mit dem LLM selbst, sondern vor allem mit dem, was und wie es an Kontext übergeben bekommt.

Die vorgestellten Pre- und Post-Retrieval-Techniken sind keine bloßen Feinjustierungen, sondern entscheidende Stellschrauben in der RAG-Pipeline. Sie helfen, semantisch relevantere Inhalte zu identifizieren, Kontextüberfrachtung zu vermeiden und Modellschwächen – wie etwa das "Lost-in-the-Middle"-Phänomen – gezielt zu kompensieren.

Advanced RAG ist somit kein Selbstläufer, sondern erfordert bewusstes Engineering entlang der gesamten Pipeline – von der Dokumentvorverarbeitung bis hin zur Reihenfolge der Prompt-Chunks.
## Take Aways
1. Chunking ist mehr als Textzerlegung: Semantisch kohärente Chunks und Metadaten verbessern Retrieval-Qualität erheblich.

2. Gute Queries sind (halb) gewonnene Antworten: Query-Rewriting und -Expansion helfen, die Nutzerintention präziser abzubilden.

3. Nicht jeder Treffer ist ein Gewinn: Nach dem Retrieval muss eine gezielte Kontextverdichtung stattfinden, um Relevanz und Fokus zu wahren.

4. Retriever-Kombination erfordert Sorgfalt: Methoden wie Reciprocal Rank Fusion können helfen – bergen aber auch Informationsverlust.

5. Kontextplatz ist kostbar: Richtiges Re-Ranking unter Berücksichtigung von LLM-Verarbeitungsschwächen (z. B. "Lost-in-the-Middle" steigert die Antwortqualität messbar.

