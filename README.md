
<img src="https://github.com/natasha/natasha-logos/blob/master/slovnet.svg">

![CI](https://github.com/natasha/slovnet/actions/workflows/test.yml/badge.svg)

SlovNet is a Python library for deep-learning based NLP modeling for Russian language. Library is integrated with other <a href="https://github.com/natasha/">Natasha</a> projects: <a href="https://github.com/natasha/nerus">Nerus</a> — large automatically annotated corpus, <a href="https://github.com/natasha/razdel">Razdel</a> — sentence segmenter, tokenizer and <a href="https://github.com/natasha/navec">Navec</a> — compact Russian embeddings. Slovnet provides high quality practical models for Russian NER, morphology and syntax, see <a href="#evaluation">evaluation section</a> for more:

* NER is 1-2% worse than current BERT SOTA by DeepPavlov but 60 times smaller in size (~30 MB) and works fast on CPU (~25 news articles/sec).
* Morphology tagger and syntax parser have comparable accuracy on news dataset with large SOTA BERT models, take 50 times less space (~30 MB), work faster on CPU (~500 sentences/sec). 

## Downloads

<table>

<tr>
<th>Model</th>
<th>Size</th>
<th>Description</th>
</tr>

<tr>
<td>
  <a href="https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar">slovnet_ner_news_v1.tar</a>
</td>
<td>2MB</td>
<td>
  Russian NER, standart PER, LOC, ORG annotation, trained on news articles.
</td>
</tr>

<tr>
<td>
  <a href="https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_morph_news_v1.tar">slovnet_morph_news_v1.tar</a>
</td>
<td>2MB</td>
<td>
  Russian morphology tagger optimized for news articles.
</td>
</tr>

<tr>
<td>
  <a href="https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar">slovnet_syntax_news_v1.tar</a>
</td>
<td>3MB</td>
<td>
  Russian syntax parser optimized for news articles.
</td>
</tr>

</table>

## Install

During inference Slovnet depends only on Numpy. Library supports Python 3.5+, PyPy 3.

```bash
$ pip install slovnet
```

## Usage

Download model weights and vocabs package, use links from <a href="#downloads">downloads section</a> and <a href="https://github.com/natasha/navec#downloads">Navec download section</a>. Optionally install <a href="https://github.com/natasha/ipymarkup">Ipymarkup</a> to visualize NER markup.

Slovnet annotator `map` method has list of items as input and same size iterator over markups as output. Internally items are processed in batches of size `batch_size`. Default size is 8, larger batch — more RAM, better CPU utilization. `__call__` method just calls `map` with a list of 1 item.

### NER

```python
>>> from navec import Navec
>>> from slovnet import NER
>>> from ipymarkup import show_span_ascii_markup as show_markup

>>> text = 'Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза. В новом списке фигурирует Леонид Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там присутствуют Владимир Бидевка и Денис Мирошниченко, председатели законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена Кравченко, председатели ЦИК обеих республик. Выборы прошли в непризнанных республиках Донбасса 11 ноября. На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР. Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после встречи с украинским лидером Петром Порошенко осудили проведение выборов, заявив, что они нелегитимны и «подрывают территориальную целостность и суверенитет Украины». Позже к осуждению присоединились США с обещаниями новых санкций для России.'

>>> navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
>>> ner = NER.load('slovnet_ner_news_v1.tar')
>>> ner.navec(navec)

>>> markup = ner(text)
>>> show_markup(markup.text, markup.spans)
Европейский союз добавил в санкционный список девять политических 
LOC─────────────                                                  
деятелей из самопровозглашенных республик Донбасса — Донецкой народной
                                          LOC─────   LOC──────────────
 республики (ДНР) и Луганской народной республики (ЛНР) — в связи с 
─────────────────   LOC────────────────────────────────             
прошедшими там выборами. Об этом говорится в документе, опубликованном
 в официальном журнале Евросоюза. В новом списке фигурирует Леонид 
                       LOC──────                            PER────
Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там 
────────                                        LOC                  
присутствуют Владимир Бидевка и Денис Мирошниченко, председатели 
             PER─────────────   PER───────────────               
законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена 
                        LOC   LOC          PER─────────────   PER───
Кравченко, председатели ЦИК обеих республик. Выборы прошли в 
─────────               ORG                                  
непризнанных республиках Донбасса 11 ноября. На них удержали лидерство
                         LOC─────                                     
 действующие руководители и партии — Денис Пушилин и «Донецкая 
                                     PER──────────    ORG──────
республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.
──────────    LOC   PER────────────              ORG──────────    LOC 
 Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после
           LOC──── PER─────────────           LOC PER───────────      
 встречи с украинским лидером Петром Порошенко осудили проведение 
                              PER─────────────                    
выборов, заявив, что они нелегитимны и «подрывают территориальную 
целостность и суверенитет Украины». Позже к осуждению присоединились 
                          LOC────                                    
США с обещаниями новых санкций для России.
LOC                                LOC─── 

```

### Morphology

Morphology annotator processes tokenized text. To split the input into sentencies and tokens use <a href="https://github.com/natasha/razdel">Razdel</a>.

```python
>>> from razdel import sentenize, tokenize
>>> from navec import Navec
>>> from slovnet import Morph

>>> chunk = []
>>> for sent in sentenize(text):
>>>     tokens = [_.text for _ in tokenize(sent.text)]
>>>     chunk.append(tokens)
>>> chunk[:1]
[['Европейский', 'союз', 'добавил', 'в', 'санкционный', 'список', 'девять', 'политических', 'деятелей', 'из', 'самопровозглашенных', 'республик', 'Донбасса', '—', 'Донецкой', 'народной', 'республики', '(', 'ДНР', ')', 'и', 'Луганской', 'народной', 'республики', '(', 'ЛНР', ')', '—', 'в', 'связи', 'с', 'прошедшими', 'там', 'выборами', '.']]

>>> navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
>>> morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
>>> morph.navec(navec)

>>> markup = next(morph.map(chunk))
>>> for token in markup.tokens:
>>>     print(f'{token.text:>20} {token.tag}')
         Европейский ADJ|Case=Nom|Degree=Pos|Gender=Masc|Number=Sing
                союз NOUN|Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing
             добавил VERB|Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act
                   в ADP
         санкционный ADJ|Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing
              список NOUN|Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing
              девять NUM|Case=Nom
        политических ADJ|Case=Gen|Degree=Pos|Number=Plur
            деятелей NOUN|Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur
                  из ADP
 самопровозглашенных ADJ|Case=Gen|Degree=Pos|Number=Plur
           республик NOUN|Animacy=Inan|Case=Gen|Gender=Fem|Number=Plur
            Донбасса PROPN|Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing
                   — PUNCT
            Донецкой ADJ|Case=Gen|Degree=Pos|Gender=Fem|Number=Sing
            народной ADJ|Case=Gen|Degree=Pos|Gender=Fem|Number=Sing
          республики NOUN|Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing
                   ( PUNCT
                 ДНР PROPN|Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing
                   ) PUNCT
                   и CCONJ
           Луганской ADJ|Case=Gen|Degree=Pos|Gender=Fem|Number=Sing
            народной ADJ|Case=Gen|Degree=Pos|Gender=Fem|Number=Sing
          республики NOUN|Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing
                   ( PUNCT
                 ЛНР PROPN|Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing
                   ) PUNCT
                   — PUNCT
                   в ADP
               связи NOUN|Animacy=Inan|Case=Loc|Gender=Fem|Number=Sing
                   с ADP
          прошедшими VERB|Aspect=Perf|Case=Ins|Number=Plur|Tense=Past|VerbForm=Part|Voice=Act
                 там ADV|Degree=Pos
            выборами NOUN|Animacy=Inan|Case=Ins|Gender=Masc|Number=Plur
                   . PUNCT

```

### Syntax

Syntax parser processes sentencies split into tokens. Use <a href="https://github.com/natasha/razdel">Razdel</a> for segmentation.

```python
>>> from ipymarkup import show_dep_ascii_markup as show_markup
>>> from razdel import sentenize, tokenize
>>> from navec import Navec
>>> from slovnet import Syntax

>>> chunk = []
>>> for sent in sentenize(text):
>>>     tokens = [_.text for _ in tokenize(sent.text)]
>>>     chunk.append(tokens)
>>> chunk[:1]
[['Европейский', 'союз', 'добавил', 'в', 'санкционный', 'список', 'девять', 'политических', 'деятелей', 'из', 'самопровозглашенных', 'республик', 'Донбасса', '—', 'Донецкой', 'народной', 'республики', '(', 'ДНР', ')', 'и', 'Луганской', 'народной', 'республики', '(', 'ЛНР', ')', '—', 'в', 'связи', 'с', 'прошедшими', 'там', 'выборами', '.']]

>>> navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
>>> syntax = Syntax.load('slovnet_syntax_news_v1.tar')
>>> syntax.navec(navec)

>>> markup = next(syntax.map(chunk))

# Convert CoNLL-style format to source, target indices
>>> words, deps = [], []
>>> for token in markup.tokens:
>>>     words.append(token.text)
>>>     source = int(token.head_id) - 1
>>>     target = int(token.id) - 1
>>>     if source > 0 and source != target:  # skip root, loops
>>>         deps.append([source, target, token.rel])
>>> show_markup(words, deps)
              ┌► Европейский         amod
            ┌►└─ союз                nsubj
┌───────┌─┌─└─── добавил             
│       │ │ ┌──► в                   case
│       │ │ │ ┌► санкционный         amod
│       │ └►└─└─ список              obl
│       │   ┌──► девять              nummod:gov
│       │   │ ┌► политических        amod
│ ┌─────└►┌─└─└─ деятелей            obj
│ │       │ ┌──► из                  case
│ │       │ │ ┌► самопровозглашенных amod
│ │       └►└─└─ республик           nmod
│ │         └──► Донбасса            nmod
│ │ ┌──────────► —                   punct
│ │ │       ┌──► Донецкой            amod
│ │ │       │ ┌► народной            amod
│ │ │ ┌─┌─┌─└─└─ республики          
│ │ │ │ │ │   ┌► (                   punct
│ │ │ │ │ └►┌─└─ ДНР                 parataxis
│ │ │ │ │   └──► )                   punct
│ │ │ │ │ ┌────► и                   cc
│ │ │ │ │ │ ┌──► Луганской           amod
│ │ │ │ │ │ │ ┌► народной            amod
│ │ └─│ └►└─└─└─ республики          conj
│ │   │       ┌► (                   punct
│ │   └────►┌─└─ ЛНР                 parataxis
│ │         └──► )                   punct
│ │     ┌──────► —                   punct
│ │     │ ┌►┌─┌─ в                   case
│ │     │ │ │ └► связи               fixed
│ │     │ │ └──► с                   fixed
│ │     │ │ ┌►┌─ прошедшими          acl
│ │     │ │ │ └► там                 advmod
│ └────►└─└─└─── выборами            nmod
└──────────────► .                   punct

```

## Documentation

Materials are in Russian:

* <a href="https://natasha.github.io/ner">Article about distillation and quantization in Slovnet</a> 
* <a href="https://youtu.be/-7XT_U6hVvk?t=2034">Slovnet section of Datafest 2020 talk</a>

## Evaluation

In addition to quality metrics we measure speed and models size, parameters that are important in production:

* `init` — time between system launch and first response. It is convenient for testing and devops to have model that starts quickly.
* `disk` — file size of artefacts one needs to download before using the system: model weights, embeddings, binaries, vocabs. It is convenient to deploy compact models in production.
* `ram` — average CPU/GPU RAM usage.
* `speed` — number of input items processed per second: news articles, tokenized sentencies.

### NER

4 datasets are used for evaluation: <a href="https://github.com/natasha/corus#load_factru"><code>factru</code></a>, <a href="https://github.com/natasha/corus#load_gareev"><code>gareev</code></a>, <a href="https://github.com/natasha/corus#load_ne5"><code>ne5</code></a> and <a href="https://github.com/natasha/corus#load_bsnlp"><code>bsnlp</code></a>. Slovnet is compared to <a href="https://github.com/natasha/naeval#deeppavlov_ner"><code>deeppavlov</code></a>, <a href="https://github.com/natasha/naeval#deeppavlov_bert_ner"><code>deeppavlov_bert</code></a>, <a href="https://github.com/natasha/naeval#deeppavlov_slavic_bert_ner"><code>deeppavlov_slavic</code></a>, <a href="https://github.com/natasha/naeval#pullenti"><code>pullenti</code></a>, <a href="https://github.com/natasha/naeval#spacy"><code>spacy</code></a>, <a href="https://github.com/natasha/naeval#stanza"><code>stanza</code></a>, <a href="https://github.com/natasha/naeval#texterra"><code>texterra</code></a>, <a href="https://github.com/natasha/naeval#tomita"><code>tomita</code></a>, <a href="https://github.com/natasha/naeval#mitie"><code>mitie</code></a>.

For every column top 3 results are highlighted:

<!--- ner1 --->
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">factru</th>
      <th colspan="2" halign="left">gareev</th>
      <th colspan="3" halign="left">ne5</th>
      <th colspan="3" halign="left">bsnlp</th>
    </tr>
    <tr>
      <th>f1</th>
      <th>PER</th>
      <th>LOC</th>
      <th>ORG</th>
      <th>PER</th>
      <th>ORG</th>
      <th>PER</th>
      <th>LOC</th>
      <th>ORG</th>
      <th>PER</th>
      <th>LOC</th>
      <th>ORG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td><b>0.959</b></td>
      <td><b>0.915</b></td>
      <td><b>0.825</b></td>
      <td><b>0.977</b></td>
      <td><b>0.899</b></td>
      <td><b>0.984</b></td>
      <td><b>0.973</b></td>
      <td><b>0.951</b></td>
      <td>0.944</td>
      <td>0.834</td>
      <td>0.718</td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td><b>0.973</b></td>
      <td><b>0.928</b></td>
      <td><b>0.831</b></td>
      <td><b>0.991</b></td>
      <td><b>0.911</b></td>
      <td><b>0.996</b></td>
      <td><b>0.989</b></td>
      <td><b>0.976</b></td>
      <td><b>0.960</b></td>
      <td>0.838</td>
      <td><b>0.733</b></td>
    </tr>
    <tr>
      <th>deeppavlov</th>
      <td>0.910</td>
      <td>0.886</td>
      <td>0.742</td>
      <td>0.944</td>
      <td>0.798</td>
      <td>0.942</td>
      <td>0.919</td>
      <td>0.881</td>
      <td>0.866</td>
      <td>0.767</td>
      <td>0.624</td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td><b>0.971</b></td>
      <td><b>0.928</b></td>
      <td><b>0.825</b></td>
      <td><b>0.980</b></td>
      <td><b>0.916</b></td>
      <td><b>0.997</b></td>
      <td><b>0.990</b></td>
      <td><b>0.976</b></td>
      <td><b>0.954</b></td>
      <td><b>0.840</b></td>
      <td><b>0.741</b></td>
    </tr>
    <tr>
      <th>deeppavlov_slavic</th>
      <td>0.956</td>
      <td>0.884</td>
      <td>0.714</td>
      <td>0.976</td>
      <td>0.776</td>
      <td>0.984</td>
      <td>0.817</td>
      <td>0.761</td>
      <td><b>0.965</b></td>
      <td><b>0.925</b></td>
      <td><b>0.831</b></td>
    </tr>
    <tr>
      <th>pullenti</th>
      <td>0.905</td>
      <td>0.814</td>
      <td>0.686</td>
      <td>0.939</td>
      <td>0.639</td>
      <td>0.952</td>
      <td>0.862</td>
      <td>0.683</td>
      <td>0.900</td>
      <td>0.769</td>
      <td>0.566</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td>0.901</td>
      <td>0.886</td>
      <td>0.765</td>
      <td>0.970</td>
      <td>0.883</td>
      <td>0.967</td>
      <td>0.928</td>
      <td>0.918</td>
      <td>0.919</td>
      <td>0.823</td>
      <td>0.693</td>
    </tr>
    <tr>
      <th>stanza</th>
      <td>0.943</td>
      <td>0.865</td>
      <td>0.687</td>
      <td>0.953</td>
      <td>0.827</td>
      <td>0.923</td>
      <td>0.753</td>
      <td>0.734</td>
      <td>0.938</td>
      <td><b>0.838</b></td>
      <td>0.724</td>
    </tr>
    <tr>
      <th>texterra</th>
      <td>0.900</td>
      <td>0.800</td>
      <td>0.597</td>
      <td>0.888</td>
      <td>0.561</td>
      <td>0.901</td>
      <td>0.777</td>
      <td>0.594</td>
      <td>0.858</td>
      <td>0.783</td>
      <td>0.548</td>
    </tr>
    <tr>
      <th>tomita</th>
      <td>0.929</td>
      <td></td>
      <td></td>
      <td>0.921</td>
      <td></td>
      <td>0.945</td>
      <td></td>
      <td></td>
      <td>0.881</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>mitie</th>
      <td>0.888</td>
      <td>0.861</td>
      <td>0.532</td>
      <td>0.849</td>
      <td>0.452</td>
      <td>0.753</td>
      <td>0.642</td>
      <td>0.432</td>
      <td>0.736</td>
      <td>0.801</td>
      <td>0.524</td>
    </tr>
  </tbody>
</table>
<!--- ner1 --->

`it/s` — news articles per second, 1 article ≈ 1KB.

<!--- ner2 --->
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>init, s</th>
      <th>disk, mb</th>
      <th>ram, mb</th>
      <th>speed, it/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td><b>1.0</b></td>
      <td><b>27</b></td>
      <td><b>205</b></td>
      <td>25.3</td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td>5.0</td>
      <td>473</td>
      <td>9500</td>
      <td><b>40.0 (gpu)</b></td>
    </tr>
    <tr>
      <th>deeppavlov</th>
      <td>5.9</td>
      <td>1024</td>
      <td>3072</td>
      <td>24.3 (gpu)</td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td>34.5</td>
      <td>2048</td>
      <td>6144</td>
      <td>13.1 (gpu)</td>
    </tr>
    <tr>
      <th>deeppavlov_slavic</th>
      <td>35.0</td>
      <td>2048</td>
      <td>4096</td>
      <td>8.0 (gpu)</td>
    </tr>
    <tr>
      <th>pullenti</th>
      <td><b>2.9</b></td>
      <td><b>16</b></td>
      <td><b>253</b></td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td>8.0</td>
      <td>140</td>
      <td>625</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>stanza</th>
      <td>3.0</td>
      <td>591</td>
      <td>11264</td>
      <td>3.0 (gpu)</td>
    </tr>
    <tr>
      <th>texterra</th>
      <td>47.6</td>
      <td>193</td>
      <td>3379</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>tomita</th>
      <td><b>2.0</b></td>
      <td><b>64</b></td>
      <td><b>63</b></td>
      <td><b>29.8</b></td>
    </tr>
    <tr>
      <th>mitie</th>
      <td>28.3</td>
      <td>327</td>
      <td>261</td>
      <td><b>32.8</b></td>
    </tr>
  </tbody>
</table>
<!--- ner2 --->

### Morphology

<a href="https://github.com/natasha/corus#load_gramru">Datasets from GramEval2020</a> are used for evaluation:

* `news` — sample from Lenta.ru.
* `wiki` — UD GSD.
* `fiction` — SynTagRus + JZ.
* `social`, `poetry` — social, poetry subset of Taiga.

Slovnet is compated to a number of existing morphology taggers: <a href="https://github.com/natasha/naeval#deeppavlov_morph"><code>deeppavlov</code></a>, <a href="https://github.com/natasha/naeval#deeppavlov_bert_morph"><code>deeppavlov_bert</code></a>, <a href="https://github.com/natasha/naeval#rupostagger"><code>rupostagger</code></a>, <a href="https://github.com/natasha/naeval#rnnmorph"><code>rnnmorph</code></a>, <a href="https://github.com/natasha/naeval#mary"><code>maru</code></a>, <a href="https://github.com/natasha/naeval#udpipe"><code>udpipe</code></a>, <a href="https://github.com/natasha/naeval#spacy"><code>spacy</code></a>, <a href="https://github.com/natasha/naeval#stanza"><code>stanza</code></a>.

For every column top 3 results are highlighted. `slovnet` was trained only on news dataset:

<!--- morph1 --->
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>news</th>
      <th>wiki</th>
      <th>fiction</th>
      <th>social</th>
      <th>poetry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td><b>0.961</b></td>
      <td>0.815</td>
      <td>0.905</td>
      <td>0.807</td>
      <td>0.664</td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td><b>0.982</b></td>
      <td><b>0.884</b></td>
      <td><b>0.990</b></td>
      <td><b>0.890</b></td>
      <td><b>0.856</b></td>
    </tr>
    <tr>
      <th>deeppavlov</th>
      <td>0.940</td>
      <td>0.841</td>
      <td>0.944</td>
      <td>0.870</td>
      <td><b>0.857</b></td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td>0.951</td>
      <td><b>0.868</b></td>
      <td><b>0.964</b></td>
      <td><b>0.892</b></td>
      <td><b>0.865</b></td>
    </tr>
    <tr>
      <th>udpipe</th>
      <td>0.918</td>
      <td>0.811</td>
      <td><b>0.957</b></td>
      <td>0.870</td>
      <td>0.776</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td><b>0.964</b></td>
      <td><b>0.849</b></td>
      <td>0.942</td>
      <td>0.857</td>
      <td>0.784</td>
    </tr>
    <tr>
      <th>stanza</th>
      <td>0.934</td>
      <td>0.831</td>
      <td>0.940</td>
      <td><b>0.873</b></td>
      <td>0.825</td>
    </tr>
    <tr>
      <th>rnnmorph</th>
      <td>0.896</td>
      <td>0.812</td>
      <td>0.890</td>
      <td>0.860</td>
      <td>0.838</td>
    </tr>
    <tr>
      <th>maru</th>
      <td>0.894</td>
      <td>0.808</td>
      <td>0.887</td>
      <td>0.861</td>
      <td>0.840</td>
    </tr>
    <tr>
      <th>rupostagger</th>
      <td>0.673</td>
      <td>0.645</td>
      <td>0.661</td>
      <td>0.641</td>
      <td>0.636</td>
    </tr>
  </tbody>
</table>
<!--- morph1 --->

`it/s` — sentences per second.

<!--- morph2 --->
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>init, s</th>
      <th>disk, mb</th>
      <th>ram, mb</th>
      <th>speed, it/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td><b>1.0</b></td>
      <td><b>27</b></td>
      <td><b>115</b></td>
      <td><b>532.0</b></td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td>5.0</td>
      <td>475</td>
      <td>8087</td>
      <td><b>285.0 (gpu)</b></td>
    </tr>
    <tr>
      <th>deeppavlov</th>
      <td><b>4.0</b></td>
      <td>32</td>
      <td>10240</td>
      <td>90.0 (gpu)</td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td>20.0</td>
      <td>1393</td>
      <td>8704</td>
      <td>85.0 (gpu)</td>
    </tr>
    <tr>
      <th>udpipe</th>
      <td>6.9</td>
      <td>45</td>
      <td><b>242</b></td>
      <td>56.2</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td>8.0</td>
      <td>140</td>
      <td>579</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>stanza</th>
      <td><b>2.0</b></td>
      <td>591</td>
      <td>393</td>
      <td><b>92.0</b></td>
    </tr>
    <tr>
      <th>rnnmorph</th>
      <td>8.7</td>
      <td><b>10</b></td>
      <td>289</td>
      <td>16.6</td>
    </tr>
    <tr>
      <th>maru</th>
      <td>15.8</td>
      <td>44</td>
      <td>370</td>
      <td>36.4</td>
    </tr>
    <tr>
      <th>rupostagger</th>
      <td>4.8</td>
      <td><b>3</b></td>
      <td><b>118</b></td>
      <td>48.0</td>
    </tr>
  </tbody>
</table>
<!--- morph2 --->

### Syntax

Slovnet is compated to several existing syntax parsers: <a href="https://github.com/natasha/naeval#udpipe"><code>udpipe</code></a>, <a href="https://github.com/natasha/naeval#spacy"><code>spacy</code></a>, <a href="https://github.com/natasha/naeval#deeppavlov_bert_syntax"><code>deeppavlov</code></a>, <a href="https://github.com/natasha/naeval#stanza"><code>stanza</code></a>.

<!--- syntax1 --->
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">news</th>
      <th colspan="2" halign="left">wiki</th>
      <th colspan="2" halign="left">fiction</th>
      <th colspan="2" halign="left">social</th>
      <th colspan="2" halign="left">poetry</th>
    </tr>
    <tr>
      <th></th>
      <th>uas</th>
      <th>las</th>
      <th>uas</th>
      <th>las</th>
      <th>uas</th>
      <th>las</th>
      <th>uas</th>
      <th>las</th>
      <th>uas</th>
      <th>las</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td>0.907</td>
      <td>0.880</td>
      <td>0.775</td>
      <td>0.718</td>
      <td>0.806</td>
      <td>0.776</td>
      <td>0.726</td>
      <td>0.656</td>
      <td>0.542</td>
      <td>0.469</td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td><b>0.965</b></td>
      <td><b>0.936</b></td>
      <td><b>0.891</b></td>
      <td><b>0.828</b></td>
      <td><b>0.958</b></td>
      <td><b>0.940</b></td>
      <td><b>0.846</b></td>
      <td><b>0.782</b></td>
      <td><b>0.776</b></td>
      <td><b>0.706</b></td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td><b>0.962</b></td>
      <td><b>0.910</b></td>
      <td><b>0.882</b></td>
      <td><b>0.786</b></td>
      <td><b>0.963</b></td>
      <td><b>0.929</b></td>
      <td><b>0.844</b></td>
      <td><b>0.761</b></td>
      <td><b>0.784</b></td>
      <td><b>0.691</b></td>
    </tr>
    <tr>
      <th>udpipe</th>
      <td>0.873</td>
      <td>0.823</td>
      <td>0.622</td>
      <td>0.531</td>
      <td>0.910</td>
      <td>0.876</td>
      <td>0.700</td>
      <td>0.624</td>
      <td>0.625</td>
      <td>0.534</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td><b>0.943</b></td>
      <td><b>0.916</b></td>
      <td><b>0.851</b></td>
      <td><b>0.783</b></td>
      <td>0.901</td>
      <td>0.874</td>
      <td><b>0.804</b></td>
      <td><b>0.737</b></td>
      <td>0.704</td>
      <td><b>0.616</b></td>
    </tr>
    <tr>
      <th>stanza</th>
      <td>0.940</td>
      <td>0.886</td>
      <td>0.815</td>
      <td>0.716</td>
      <td><b>0.936</b></td>
      <td><b>0.895</b></td>
      <td>0.802</td>
      <td>0.714</td>
      <td><b>0.713</b></td>
      <td>0.613</td>
    </tr>
  </tbody>
</table>
<!--- syntax1 --->

`it/s` — sentences per second.

<!--- syntax2 --->
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>init, s</th>
      <th>disk, mb</th>
      <th>ram, mb</th>
      <th>speed, it/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>slovnet</th>
      <td><b>1.0</b></td>
      <td><b>27</b></td>
      <td><b>125</b></td>
      <td><b>450.0</b></td>
    </tr>
    <tr>
      <th>slovnet_bert</th>
      <td><b>5.0</b></td>
      <td>504</td>
      <td>3427</td>
      <td><b>200.0 (gpu)</b></td>
    </tr>
    <tr>
      <th>deeppavlov_bert</th>
      <td>34.0</td>
      <td>1427</td>
      <td>8704</td>
      <td><b>75.0 (gpu)</b></td>
    </tr>
    <tr>
      <th>udpipe</th>
      <td>6.9</td>
      <td><b>45</b></td>
      <td><b>242</b></td>
      <td>56.2</td>
    </tr>
    <tr>
      <th>spacy</th>
      <td>9.0</td>
      <td><b>140</b></td>
      <td><b>579</b></td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>stanza</th>
      <td><b>3.0</b></td>
      <td>591</td>
      <td>890</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
<!--- syntax2 --->

## Support

- Chat — https://telegram.me/natural_language_processing
- Issues — https://github.com/natasha/slovnet/issues
- Commercial support — https://lab.alexkuk.ru

## Development

Dev env

```bash
python -m venv ~/.venvs/natasha-slovnet
source ~/.venvs/natasha-slovnet/bin/activate

pip install -r requirements/dev.txt
pip install -e .
```

Test

```bash
make test
```

Rent GPU

```bash
yc compute instance create \
  --name gpu \
  --zone ru-central1-a \
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-1804-lts-ngc,type=network-ssd,size=20 \
  --cores=8 \
  --memory=96 \
  --gpus=1 \
  --ssh-key ~/.ssh/id_rsa.pub \
  --folder-name default \
  --platform-id gpu-standard-v1 \
  --preemptible

yc compute instance delete --name gpu
```

Setup instance

```
sudo locale-gen ru_RU.UTF-8

sudo apt-get update
sudo apt-get install -y \
  python3-pip

# grpcio long install ~10m, not using prebuilt wheel
# "it is not compatible with this Python" 
sudo pip3 install -v \
  jupyter \
  tensorboard

mkdir runs
nohup tensorboard \
  --logdir=runs \
  --host=localhost \
  --port=6006 \
  --reload_interval=1 &

nohup jupyter notebook \
  --no-browser \
  --allow-root \
  --ip=localhost \
  --port=8888 \
  --NotebookApp.token='' \
  --NotebookApp.password='' &

ssh -Nf gpu -L 8888:localhost:8888 -L 6006:localhost:6006

scp ~/.slovnet.json gpu:~
rsync --exclude data -rv . gpu:~/slovnet
rsync -u --exclude data -rv 'gpu:~/slovnet/*' .
```

Intall dev

```bash
pip3 install -r slovnet/requirements/dev.txt -r slovnet/requirements/gpu.txt
pip3 install -e slovnet
```

Release

```bash
# Update setup.py version

git commit -am 'Up version'
git tag v0.6.0

git push
git push --tags

# Github Action builds dist and publishes to PyPi
```
