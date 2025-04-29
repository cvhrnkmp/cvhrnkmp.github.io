+++
 date = '2025-04-19T15:26:23+02:00'
 draft = false
 math = true
 number-sections = true
 disableComments = false
 title = 'Eine sanfte Einführung in Retrieval Augmented Generation (RAG)'
 [params]
    series = ["RAG"]
+++

"Retrieval was…?" – Ja genau, Retrieval Augmented Generation. Zu Deutsch "Abruf‑erweiterte Generierung", aber was steckt dahinter?
Die bekanntesten Chat-Oberflächen wie ChatGPT oder Claude nutzen große Sprachmodelle (LLMs), die auf enormen Datenmengen trainiert sind. Doch diese Modelle haben Grenzen:

- Sie können veraltetes oder unvollständiges Wissen enthalten.

- Sie tendieren zu Halluzinationen, wenn sie Fakten nicht verifizieren können.

- Interne Unternehmensdaten sind für sie nicht zugänglich – und bleiben es oft aus Sicherheitsgründen.

**Retrieval Augmented Generation** löst diese Herausforderungen auf elegante Weise: Das Modell bezieht in Echtzeit relevante Informationen aus externen oder internen Quellen ein, um seine Antworten auf geprüfte Fakten zu stützen. Dabei fungiert RAG als dynamischer Wissensbrunnen, der sowohl frische Daten (zum Beispiel Quartalsberichte oder aktuelle Gesetzestexte) als auch firmenspezifisches Know-how (etwa interne Leitfäden oder Protokolle) nahtlos in den Antwortprozess integriert. So profitieren Unternehmen von deutlich präziseren, stets aktuellen Ergebnissen – bei gleichzeitiger Wahrung der Vertraulichkeit sensibler Informationen. Zudem macht RAG Wissen und Informationen jederzeit verfügbar, indem es relevante Inhalte direkt bereitstellt und so den Zugriff auf entscheidungsrelevantes Fachwissen erleichtert.

**Folgendes erwartet dich:**

## Die Idee hinter RAG

Die Grundidee von RAG besteht darin, dass ein Sprachmodell nicht allein auf sein vortrainiertes Wissen angewiesen ist, sondern bei jeder Anfrage gezielt in einer umfangreichen Dokumentenbasis nach relevanten Informationen sucht. Statt Hypothesen oder Vermutungen liefert das System exakte Fakten aus Quellen wie Unternehmens-Wikis, juristischen Leitfäden oder Forschungsartikeln. Neue und sich schnell ändernde Daten – etwa Quartalsberichte oder Gesetzesänderungen – werden einfach zur Retrieval-System hinzugefügt, ohne dass das Sprachmodell neu trainiert werden muss. Gleichzeitig bleiben vertrauliche interne Dokumente sicher abgeschottet: Nur freigegebene Passagen fließen in den Kontext ein, sodass Datenschutz und Compliance jederzeit gewahrt sind. Dank spezialisierter Vektor- und Keyword-Suchsysteme skaliert RAG mühelos von kleineren Datenbeständen bis hin zu Millionen von Dokumenten.

## So funktioniert RAG (High-Level)

Als erstes muss eine Dokumentenbasis vorbereitet werden. Das heißt Datenquellen identifiezieren und sammeln. Anschließend werden die Daten bereinigt und sinnvoll in Abschnitte (»Chunks«) aufteilt.
Darauf aufbauend kann ein Retrieval-System mittels Vektoren (Embeddings) und/oder Schlüsselwörter die relevantesten Textpassagen finden und bereitstellen.
Anschließend wird die Nutzerfrage und die abgerufene Passagen zu einem gemeinsamen Prompt vereint.
Das Sprachmodell erstellt eine Antwort, die direkt auf den abgerufenen Textpassagen mit Fakten basiert.
Darüber hinaus gibt es viele Optionen - wie Reranking oder Filterung, die für eine Optimierung hin zu höchster Präzision und Verständlichkeit mobilisiert werden können.

### Vector Retrieving

Beim Vector Retrieving (auch *Dense Search* genannt) stehen semantische Ähnlichkeiten im Fokus. Die Vorgehensweise gliedert sich in drei Teilschritte. In der folgenden Grafik sind die Teilschritte im Flowchart benannt und im folgenden aufgegriffen:

![RAG-Flow](/Post2/RAG_Flow_1.excalidraw.svg)

<!--<p> Die folgende Grafik zeigt den Prozess von RAG. Zuerst werden die Dokumenteabschnitte aufgeteilt. Anschlißend wird für jeden Abschnitt ein Vektor berechnet und in eine Vektordatenbank abgespeichert. Nach dem der Anwender eine *Query* gestellt hat, kann optional eine *Query Augmentation* durchgeführt - das heißt unvollständige Queries ergänzt, um eine qualitative Anfrage zu erhalten. Damit wird die Vektor-Datenbank durchsucht und die relevantesten Dokumentenabschnitte zurückgegeben. Mit den relevantesten $k$-Abschnitte wird die eigentliche Query ergänzt, womit das Sprachmodell gepromptet wird. </p> -->

(1) Zuerst werden **Embeddings** erzeugt: Jeder Dokumentenabschnitt $d_i$ und jede Nutzerfrage $q$ wird mithilfe eines Embedding-Modells (z. B. Sentence-BERT, OpenAI-Embeddings) in einen hochdimensionalen Vektorraum eingebettet - sogenannten Dokumentenvektor $\vec{d_i}$. Ähnliche Inhalte liegen dabei im Vektorraum nahe beieinander.
Die Vektoren von Dokumentenabschnitten werden in Vektor-Datenbanken abgespeichert. Solche Systeme ermöglichen extrem schnelle Suchanfragen selbst bei sehr großen Datenmengen.

(2) Die Suche nach den relevantesten Dokumentenabschnitte baut auf die Distanz der Vektoren auf. Für eine eingehende Frage wird ihr Vektor $\vec{q}$ berechnet und in der Datenbank eine kNN-Suche ($k$-Nearest-Neighbors) ausgeführt. Das heißt die Distanz zwischen dem Anfragevektor $\vec{q}$ und den Dokumentenvektoren $\vec{d_i}$ wird berechnet. Je kleiner die Distanz desto ähnlicher ist der Inhalt. Somit werden die $k$-Vektoren mit der höchsten Ähnlichkeit - bzw. dem kleinsten Abstand -  als die relevanteste Textpassagen zurückgegeben.

(3) Anschließend werdend die relevantestes Textpassagen in ein Prompt als *Context* hinzugefügt und die Nutzeranfrage als *Query* hinzugefügt. 

### Cosine Similarity

Durchgesetzt hat sich jedoch die Cosine Similarity als das gängigste Ähnlichkeitsmaß zwischen Vektoren in Retrieval Augmented Generation. Sie berechnet den Winkel zwischen dem Fragevektor $\vec{q}$ und einem Dokumentenvektor $\vec{d_i}$ im Vektorraum:

$$
\cos(q, d_i) = \frac{q \cdot d_i}{\lVert q\rVert \cdot \lVert d_i \rVert}
$$

Ein Wert nahe 1 zeigt hohe semantische Ähnlichkeit an, ein Wert nahe Null ist neutral und ein Wert nahe -1 zeigt die große Unähnlichkeit. Cosine Similarity bietet den Vorteil, dass das Maß robust gegenüber unterschiedlichen Textlängen ist, da es sich auf die Richtung der Vektoren und nicht auf die Länge konzentriert.

### Kurzes Beispiel:

Nehmen wir an, dass der folgende Abschnitt aus der unternehmens-internen Wiki kommt und somit nicht der Öffentlichkeit zur Verfügung steht. Das heißt auch, dass die Information nicht im Trainingsdatensatz von OpenAI (ChatGPT) oder Antrophic (Claude-Modelle) für ihre Modelle enthalten ist. Desweiteren kann es der Fall sein, dass die Informationen erst nach dem Training der Modelle erstanden sind und deswegen nicht im Trainingsdatensatz enthalten ist:

``` Quartalsbericht Q1 2025: Unser Unternehmen erwirtschaftete im ersten Quartal 2025 einen Umsatz von 12,4 Mio. € und einen Nettogewinn von 1,8 Mio. €. ```

Und folgende Nutzer-Frage: 

``` "Welche finanziellen Kennzahlen weist die CVBlog AG im Q1 2025 auf?" ```

So mobilisieren wir das RAG-System:

1. Die Retrieval-Komponente identifiziert den passenden Textausschnitt aus dem Wiki.

2. Anschließend wird ein kombinierter Prompt erstellt.

> **Kontext**: Quartalsbericht Q1 2025: Die CVBlog AG erwirtschaftete im ersten Quartal 2025 einen Umsatz von 12,4 Mio. € und einen Nettogewinn von 1,8 Mio. €.

> **Frage**: Welche finanziellen Kennzahlen weist unser Unternehmen im Q1 2025 auf?

> **Promptvorlage**: 

> ```text
> Context: {abgerufener Kontext}
> Frage: {Ihre Frage} 
> Antworten Sie ausschließlich auf Grundlage des kontextuellen Abschnitts.
> ```

> **Fertiger Prompt**: 

> ```text
> Context: Quartalsbericht Q1 2025: Die CVBlog AG erwirtschaftete im ersten Quartal 2025 einen Umsatz von 12,4 Mio. € und einen Nettogewinn von 1,8 Mio. €.
> Frage: Welche finanziellen Kennzahlen weist die CVBlog AG im Q1 2025 auf? 
> Antworten Sie ausschließlich auf Grundlage des kontextuellen Abschnitts.
> ```

3. Daraufhin generiert das LLM die Antwort und nutzt diesen Kontext für eine präzise Antwort:

> „Im ersten Quartal 2025 erzielte unser Unternehmen einen Umsatz von 12,4 Mio. € und einen Nettogewinn von 1,8 Mio. €."

Dank In-Context Learning greift das Sprachmodell gezielt auf den abgerufenen Abschnitt zurück, um eine verlässliche, faktengestützte Antwort zu liefern.

Der [ChatGPT-Verlauf](https://chatgpt.com/share/6810f024-c014-8006-9eba-a131640c3297) zeigt das Ergebnis. Zuerst wurde die Frage ohne Kontext gestellt und anschließend wurde der *fertige Prompt* als Anfrage genutzt.

## Finetuning vs. RAG
"Warum macht man dann kein Finetuning von den Modellen auf bspw. firmen-internen Daten?" -  eine Frage die ich häufig höre. RAG hat viele Vorteile gegenüber dem Finetuning. Neben der schnelleren Aktualisierung der Daten und informationen, zeigt sich auch eine große Kosteneffizienz. Das Finetuning von Large Language Models ist aufwending und langwierig, dadurch entstehen hohe Kosten. Häufig besitzen Firmen und Institutionen die benötigte Rechenkapazität und müsseen Cloud-Dienste in Anspruch nehmen, sodass die internen Daten in die Cloud geladen werden müssen. Als Beispiel: Ein 70 Milliarden-Parameter großes Modell wie Llama3.3 benötigt grob 520.000 GPU-Stunden bei Kosten von etwa 1,5$/GPUh sind es rund 780.000$ (760.000€) in der Cloud - die Trainingsdauer beträgt ca. 43 Tage. Nimmt man an das die Kosten für ein Finetuning nur 10% des Pre-Trainings in Anspruch nimmt, sind es 76.000€ und 4-5 Tage Trainingsaufwand. Doch mal ehrlich, nach 4-5 Tagen sind viele Daten schon wieder alt oder neue sind dazu gekommen und wer möchte schon Tage auf ein aktuelles Modell warten und jede Woche 76.000€ ausgeben?

Außerdem ist die Modularität in RAG-Systemen höher. Das Sprachmodell zur Generierung der Antwort kann ohne großen Aufwand ausgetauscht werden, soadss immer mit den leistungsstärksten State-of-the-Art Modellen gearbeitet wird. Wird hingegen ein Modell trainiert, muss abgewägt werden, ob es sich betriebswirtschaftlich lohnt, da das neue Modell wieder auf die unternehmens-internen Daten trainiert werden muss. 

Ein großer Vorteil der immer wieder für Interesse sorgt, sind Referenzangaben, worauf die Antwort des Sprachmodels basiert. Das steigert die Akzeptantz und das Verständnis der Antwort bei Anwendern und ist häufig der Grund, warum sich Unternehmen für RAG entscheiden.

Alles im allem bleibt überwiegen die Vorteile von RAG gegenüber Finetuning: Schnelle Aktualisierung der Daten, Kosteneffizient, Datenschutz und Complaince, Modularität und Flexibilität und Referenzangaben zur Quelle der Antwort.

## Typische Anwendungsfälle

Typische Use Cases sind eine schnelle Suche in unternehmens-internen **Wikis, Handbüchern und Protokollen**. Auch **Kundenservice**-Chatbots können automatisiert auf Basis von Unternehmensrichtlinien spezifische Antworten geben. Durch nachvollziehbare Antworten mit Verweis auf konkrete Gesetze oder Vorschriften kann RAG **Compliance und Audit** Prozesse effizienter und effektiver machen.
Im Healthcare & Life Science Bereich können **evidenzbasierte Zusammenfassungen** aus aktuellen Studien und Leitlinien erstellt werden. In der **Marktforschung und Business Intelligence** werden aktuelle Reports und Analysen als Datenquelle für Chat-Anwendungen nutzbar.

## Zusammenfassung

Dieser Beitrag stellt dar, wie Sprachmodelle durch Retrieval Augmented Generation um Echtzeit-Wissen ergänzt wird und so präzise, aktuelle und verlässliche Antworten gewährleistet. Es wurden die drei Hauptkomponenten - Retrieval, Generation und Augmentation - vorstellt und dargestellt, wie Vector Retrieving mittels Cosine Similarity relevante Dokumentenabschnitte punktgenau identifiziert. Ein Minimalbeispiel ohne Code zeigte, wie RAG mittel In-Context Learning das Modell befähigt, abgerufene Informationen direkt zu nutzen.

### Takeaways

1. RAG kann dynamisch externe Dokumente integrieren und macht Wissen jederzeit verfügbar. Dadurch Halluzinationen reduziert und die Faktizität/Sachlichkeit gesteigert.
2. Vorteile von RAG gegenüber Finetuning: Schnelle Aktualisierung der Daten, Kosteneffizient, Datenschutz und Complaince, Modularität und Flexibilität. 
3. RAG kann eine große Effizienz- und Effektivitätssteigerungen in Unternehmensprozessen sein.
