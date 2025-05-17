+++
 date = '2025-04-30T15:26:23+02:00'
 draft = false
 math = true
 number-sections = true
 disableComments = false
 title = '#2 Retriever: Das Herz von Retrieval Augmented Generation'
+++

Das Herz von Retrieval Augmented Generation ist der Retriever, welcher die relevantesten bzw. ähnlichsten Textpassage zu einer Anfrage finden soll. *Soll* - ja richtig gehört, dass ist nicht immer der Fall. Retriever haben nicht nur Stärken, sondern auch Schwächen. Deswegen wurden im Laufe der Zeit verschiedene Retriever-Methoden entwickelt und davon haben sich zwei im Wesentlichen durchgesetzt: Vektor-Suche und Keyword-Suche. Diese beiden Verfahren können jetzt unterschiedlich miteinander kombiniert und unterschiedlich ausgeführt werden.   

Der folgende Beitrag stellt die grundlegenden Retriever-Methoden und die Möglichkeit der Kombination vor. 

## Vektor-Suche
Die Vektor-Suche basiert auf die Semantik von Text. Semantik ist einfach gesagt, die Bedeutung. Das heißt, wenn zwei Texte eine hohe semantische Ähnlichkeit haben, dann ist die Bedeutung nahezu die Gleiche. Andererseits bedeutet eine hohe semantische Ungleichheit, eine unterschiedliche Bedeutung. 

**Beispiel:**
1. In dem Satz *Ich gehe zur Bank* bezieht sich das Wort *Bank* auf ein Geldinstitut.
2. In dem Satz *Ich setze mich auf die Bank* bezieht sich das Wort *Bank* auf eine Sitzgelegenheit.

Das Beispiel zeigt die Mehrdeutigkeit von dem Wort *Bank*. Die Semantik stellt die Bedeutung des Wortes dar, wie es im Kontext verstanden wird. Ohne semantische Kenntnisse können Missverständnisse auftreten. Klarheit und Präzision werden durch die Semantik in der Sprache hergestellt, sodass Informationen in einem gegebenen Kontext verstanden werden. 

### Prozess
Bei der Vektor-Suche werden als erstes alle Dokumentenabschnitte in die Vektor-Repräsentation $D \in \\{d_0, d_1, ..., d_{n-1}, d_n \\}$ gebracht, wobei $d_i$ ein Dokumentenabschnitt entspricht und $n$ die Menge aller Dokumentenabschnitt repräsentiert. Anschließend wird jede Anfrage (Query) in einen Vektor $q$ umgewandelt. Somit stellen die Vektoren von $D$ die Dokumentenabschnitte als Punkt in einem Raum dar - gleiches gilt auch für den Vektor $q$ der Anfrage. Je kleiner die Distanz zwischen zwei Punkten ist, desto ähnlicher ist der Inhalt, welcher durch die Vektoren repräsentiert werden - unabhängig ob es sich um die Distanz zwischen zwei Dokumentenabschnitten aus $D$ handelt oder um die Distanz zwischen einem Vektor $d_i$ aus $D$ und den Anfragevektor $q$ handelt.

Die große Quizfrage ist: Wie wird die Distanz gemessen?

Die folgende Abbildung zeigt drei unterschiedliche Verfahren, um die Distanz zwischen zwei Vektoren zu ermitteln.

![Vector distance metrics](/2_Retriever/Similarity_metrics_qdrant.png)
<p style="text-align: center;">Image by <a href = 'https://qdrant.tech/blog/what-is-vector-similarity/'> Qdrant </a></p>

Dabei finden im NLP-Umfeld das *Dot Product* und die *Cosine Similarity* die häufigste Anwendung. Das Dot-Product wird oft für $k$-nearest-neighbor Aufgaben genutzt, um die $k$ naheliegendsten Punkte zu finden. Gleiches kann auf die Euclidean Distance übertragen werden. Die Cosine Similarity findet in Information Retrieval Systemen wie RAG ihre Anwendung.


Für die **Euclidean Distance** gilt:

$$ 
d(q, d_i) = \sqrt{\sum_{j=1}^{n}(d_{ij} - q_j)^2}
$$
        wobei $n$ die Dimension der Vektoren darstellt.

Für das **Dot Product** bzw. **Inner Product** gilt:

$$
\begin{equation*}
\begin{aligned}
d(q, d_i) &= q \cdot d_i \newline
&= \sum_{j=1}^{n}{q_j \cdot d_{ij}}
\end{aligned}
\end{equation*}
$$
        wobei $n$ die Dimension der Vektoren darstellt. 

Für die **Cosine Similarity** gilt:

$$
cos(q, d_i) = \frac{q \cdot d_i}{|q| \cdot |d_i|}
$$

Wie der aufmerksame Leser wahrscheinlich schon festgestellt hat, ist die Cosine Similarity die normalisierte Distanz des Inner Products. Damit ist die Anwendung der Cosine Similarity in Szenarien mit unterschiedlichen langen Vektoren empfehlenswert, da es eine bessere Aussagekraft als das Inner Product gibt.

## Keyword Suche
Im Vergleich zur Vektor-Suche konzentriert sich die Keyword-Suche vollständig auf den Inhalt ohne deren semantische Repräsentation zu betrachten. Das heißt, die Suche hebt Stichwörter aus der Query hervor, welche große Relevanz haben und damit auch in den Dokumentenabschnitten eine wesentliche Rolle spielen. Dabei hat sich der [BM25 Algorithmus von Okapi](https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf) durchgesetzt:
$$
\begin{equation*}
\text{score}(d_i,q) = \epsilon \cdot \sum_{j=1}^{n} \text{IDF}(q_j) \cdot \frac{f(q_j, d_i) \cdot (k_1 + 1)}{f(q_j, d_i) + k_1 \cdot \left(1 - b + b \cdot \frac{l(d_i)}{\text{avg(l(D))}}\right)}
\end{equation*}
$$

mit $\text{IDF}$ als *inverse term-frequency* - zu deutsch inverse Begriffshäufigkeit - von dem $j$-ten Wort in $q$ und $f(q_j, d)$ ist die Begriffshäufigkeit des Wortes in den Dokument $d_i$.

$\epsilon$, $k_1$ und $b$ sind Hyperparameter, welche bei Bedarf optimiert werden können - ansonsten gilt für $\epsilon = 0.25$, $k_1 = 1.5$ und $b=0.75$. 


Was kann aus der Formel interpretiert werden? 

1. Je länger der Dokumentenabschnitt ist, desto kleiner der Score. Das heißt, BM25 bestraft lange Dokumentenabschnitte.
2. Je häufiger ein Wort im Dokumentenabschnitt $d_i$ vorkommt, desto kleiner wird der Score. Somit bestraft BM25 Wörter die sehr häufig vorkommen wie *der*, *die*, *das*. 

Somit zielt der Algorithmus auf kurze Abschnitte ab und hohe Begriffsrelevanz innerhalb das Abschnittes. Das führt im besten Fall dazu, dass sehr präzise und auf den Punkt gebrachte Textabschnitte mit den höchsten Score gewertet werden und im Kontext des Prompts bereitgestellt werden. Daraufhin kann das LLM eine gute Aussage treffen, da häufig mit langen und unpräzisen Textabschnitten irrelevanter Inhalt zurückgegeben wird. 
  
## Hybride Suche
Die hybride Suche nach dem besten und relevantesten Text Passagen, die die Anfrage beantworten kann, vereint die Vektor-Suche und die Keyword-Suche. Dabei werden die Retriever Methoden parallel durchgeführt. Anschließend werden die Scores je Dokumentabschnitt $d_i$ zusammengefasst. Somit werden die Vorteile beider Suchverfahren kombiniert und die Schwächen geglättet. Denn die semantische Suche selbst berücksichtigt die wichtigen Schlüsselwörter aus der Query nicht, doch die Keyword-Suche. Andererseits lässt die Keyword-Suche die semantische Beziehung zwischen der Query und den Dokumentenabschnitte außer acht, doch die Vektor-Suche bildet die semantische Suche voll ab.

## HyDE Suche
Ein weiterer Schritt ist der *Hypothecial Document Embeddings* Ansatz von [Gao et. al](https://arxiv.org/pdf/2212.10496). Die Idee ist ein hypothetische Dokument zu generieren, welches die Muster eines relevanten Dokumentes erfasst und damit eine hohe Ähnlichkeit zu den geforderten/gesuchten Dokumentenabschnitte hat. Auf faktische Korrektheit wird dabei verzichtet, da es nur das Muster der gesuchten Abschnitte repräsentieren soll. Dabei wird als erstes ein LLM aufgefordert zu der Anfrage ein hypothetisches Dokument $d_h$ zu generieren, welches die Frage beantworten kann. Anstatt die Retriever mit der originalen Anfrage zu füttern, wird das hypothetische Dokument $d_h$ den Retriever Methoden zugeführt. 

## Best Practice: Hybrid HyDE Suche
[Wang et. al.](https://arxiv.org/pdf/2407.01219) hat einige Experimente durchgeführt, um die beste Kombination aus Retriever Methoden finden. Dabei stellte er unter Laborbedingungen fest, dass die Variation aus der HyDE Suche kombiniert mit der hybriden Suche, die besten Ergebnisse produziert. 

## Diskussion
*Welche Suchmethode ist denn jetzt die Beste?* - das ist abhängig vom Usecase. Denn die Vektorsuche ist stark von dem Verfahren, wie die Vektorrepräsentation der Abschnitte hergestellt wird, abhängig. Wenn dazu ein Encoder Modell verwendet wird, muss das Model, die Semantik verstehen. Da es sich häufig um interne Dokumente handelt, domain-spezifisch und nicht in englischer Sprache, kann es zur Performance einbußen kommen. Daher ist eine Kombination mit der Keysearch ein guter Weg, um Fachthermini gut abbilden zu können. 

Geht es darum schnelle Antworten zu erzeugen, muss abgewägt werden, ob der Einsatz von HyDE gewollt ist. Zum Erzeugen eines hypothetischen Dokuments ist ein LLM-Call nötig, der wiederum Zeit in Anspruch nimmt. 

## Zusammenfassung
Gemeinsam haben wir beleuchtet, wie nach den relevantesten Passagen in einem großen Datenkorpus gesucht wird. Die Methodiken zeigen Stärken und Schwächen. Eine Kombination aus den Methoden kann zu hervorragenden Synergie-Effekten führen. Für meine Projekte bedeutete es bisher immer *Testen, Testen, Testen*, um das richtige Verfahren zur Suche passender Passagen zu finden. Doch eine Frage bleibt: Wenn es im System mehrere Retriever gibt, wie wird es zu einem Ergebnis zusammengefasst?

### Take Aways

1. Es gibt viele verschiedener Retriever Methoden mit unterschiedlichen Stärken.
2. Die Retriever Methoden können beliebig kombiniert werden.
3. Es muss für jeden Use Case getestet und entschieden werden. <br>
4. Die hybride Suche ist ein guter Start, da es Synergie-Effekte mit Vektor- und Keyword-Suche schafft.

