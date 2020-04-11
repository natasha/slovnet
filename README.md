
<img src="https://github.com/natasha/natasha-logos/blob/master/slovnet.svg">

[![Build Status](https://travis-ci.org/natasha/slovnet.svg?branch=master)](https://travis-ci.org/natasha/slovnet)

SlovNet is a Python library for deep-learning based NLP modeling for Russian language. Library is integrated with other <a href="https://github.com/natasha/">Natasha</a> projects: <a href="https://github.com/natasha/nerus">large NER corpus</a> and <a href="https://github.com/natasha/navec">compact Russian embeddings</a>. SlovNet provides high quality practical model for Russian NER, it is 1-2% worse than current BERT SOTA by DeepPavlov but 60 times smaller in size (~30 MB) and works fast on CPU (~30 news articles/sec), see <a href="#evaluation">evaluation section</a> for more.

## Install

During inference `slovnet` depends only on `numpy`. Library supports Python 2.7+, 3.4+ и PyPy 3. PyPy 2 is excluded since it is hard to install `numpy` for PyPy 2.

```bash
$ pip install slovnet
```

## Usage

Download <a href="https://github.com/natasha/navec#downloads">news Navec embeddings</a> and <a href="#downloads">SlovNet news NER model</a>:

```python
>>> from navec import Navec
>>> from slovnet import NERTagger
>>> from ipymarkup import show_ascii_markup

>>> text = 'Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза. В новом списке фигурирует Леонид Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там присутствуют Владимир Бидевка и Денис Мирошниченко, председатели законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена Кравченко, председатели ЦИК обеих республик. Выборы прошли в непризнанных республиках Донбасса 11 ноября. На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР. Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после встречи с украинским лидером Петром Порошенко осудили проведение выборов, заявив, что они нелегитимны и «подрывают территориальную целостность и суверенитет Украины». Позже к осуждению присоединились США с обещаниями новых санкций для России.'

>>> navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')

>>> tagger = NERTagger.load('slovnet_ner_v1.tar', navec)
>>> markup = tagger(text)
>>> markup
SpanMarkup(
    text='Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза. В новом списке фигурирует Леонид Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там присутствуют Владимир Бидевка и Денис Мирошниченко, председатели законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена Кравченко, председатели ЦИК обеих республик. Выборы прошли в непризнанных республиках Донбасса 11 ноября. На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР. Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после встречи с украинским лидером Петром Порошенко осудили проведение выборов, заявив, что они нелегитимны и «подрывают территориальную целостность и суверенитет Украины». Позже к осуждению присоединились США с обещаниями новых санкций для России.',
    spans=[Span(
         start=0,
         stop=16,
         type='LOC'
     ), Span(
         start=108,
         stop=116,
         type='LOC'
     ), Span(
         start=119,
         stop=153,
         type='LOC'
     )
 ...
])

>>> show_ascii_markup(markup.text, markup.spans)

Европейский союз добавил в санкционный список девять политических 
LOC-------------                                                  
деятелей из самопровозглашенных республик Донбасса — Донецкой народной
                                          LOC-----   LOC--------------
 республики (ДНР) и Луганской народной республики (ЛНР) — в связи с 
-----------------   LOC--------------------------------             
прошедшими там выборами. Об этом говорится в документе, опубликованном
 в официальном журнале Евросоюза. В новом списке фигурирует Леонид 
                       LOC------                            PER----
Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там 
--------                                        LOC                  
присутствуют Владимир Бидевка и Денис Мирошниченко, председатели 
             PER-------------   PER---------------               
законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена 
                        LOC   LOC          PER-------------   PER---
Кравченко, председатели ЦИК обеих республик. Выборы прошли в 
---------               ORG                                  
непризнанных республиках Донбасса 11 ноября. На них удержали лидерство
                         LOC-----                                     
 действующие руководители и партии — Денис Пушилин и «Донецкая 
                                     PER----------    ORG------
республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.
----------    LOC   PER------------              ORG----------    LOC 
 Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после
           LOC---- PER-------------           LOC PER-----------      
 встречи с украинским лидером Петром Порошенко осудили проведение 
                              PER-------------                    
выборов, заявив, что они нелегитимны и «подрывают территориальную 
целостность и суверенитет Украины». Позже к осуждению присоединились 
                          LOC----                                    
США с обещаниями новых санкций для России.
LOC                                LOC--- 

```

## Downloads

<a href="https://github.com/natasha/slovnet/releases/download/v0.0.0/slovnet_ner_v1.tar">slovnet_ner_v1.tar</a> 1.5 MB

## Evaluation

4 datasets are used for evaluation, see <a href="https://github.com/natasha/corus">Corus</a> registry for more info: <a href="https://github.com/natasha/corus#load_factru"><code>factru</code></a>, <a href="https://github.com/natasha/corus#load_gareev"><code>gareev</code></a>, <a href="https://github.com/natasha/corus#load_ne5"><code>ne5</code></a> and <a href="https://github.com/natasha/corus#load_bsnlp"><code>bsnlp</code></a>.

`slovnet` is compared to:

* `deeppavlov` — biLSTM + CRF by DeepPavlov, see <a href="https://arxiv.org/pdf/1709.09686.pdf">their 2017 paper</a> for more.
* `deeppavlov_bert` — BERT based NER, current SOTA for Russian language, see <a href="https://www.youtube.com/watch?v=eKTA8i8s-zs">video presentation</a> describing the approach.
* <a href="http://pullenti.ru/">`pullenti`</a> — first place on factRuEval-2016, super sophisticated ruled based system.
* <a href="https://texterra.ispras.ru">`texterra`</a> — multifunctional NLP solution by <a href="https://www.ispras.ru/">ISP RAS</a>, NER is one of the features.
* <a href="https://github.com/yandex/tomita-parser/">`tomita`</a> — GLR-parser by Yandex, only grammars for `PER` are publicly available.
* <a href="https://github.com/mit-nlp/MITIE">`mitie`</a> — engine developed at MIT + <a href="http://lang.org.ua/en/models/">third party model for Russian language</a>.

For every column top 3 results are highlighted. In each case `slovnet` and `deeppavlov_bert` are 5-10% better then other systems:

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
      <td><b>0.838</b></td>
      <td><b>0.733</b></td>
    </tr>
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
      <td><b>0.944</b></td>
      <td><b>0.834</b></td>
      <td><b>0.718</b></td>
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
      <th>natasha</th>
      <td>0.867</td>
      <td>0.753</td>
      <td>0.297</td>
      <td>0.873</td>
      <td>0.347</td>
      <td>0.852</td>
      <td>0.709</td>
      <td>0.394</td>
      <td>0.836</td>
      <td>0.755</td>
      <td>0.350</td>
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

* `init` — time between system launch and first response. It is convenient for testing and devops to have model that starts quickly. `deeppavlov_bert` and `texterra` take >30 sec to start, `slovnet` takes just ~1 sec.
* `disk` — file size of artefacts one needs to download before using the system: model weights, embeddings, binaries, vocabs. It is inconvenient to deploy large models in production. `deeppavlov` models require >1 GB download, `slovnet` is just 30 MB including embeddings.
* `ram` — average memory consumption. `deeppavlov` systems and `texterra` are memory heavy, `slovnet` consumes ~200 MB of RAM.
* `speed` — number of news articles processed per second, one article is ~1 KB of text. `deeppavlov` systems process texts in batches on GPU, but they are still slover than `tomita`, `mitie` and `slovnet` that run on single CPU.

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
      <th>slovnet_bert</th>
      <td>5.0</td>
      <td>473</td>
      <td>9500</td>
      <td><b>40.0 (gpu)</b></td>
    </tr>
    <tr>
      <th>slovnet</th>
      <td><b>1.0</b></td>
      <td><b>27</b></td>
      <td>2048</td>
      <td>18.0</td>
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
      <th>pullenti</th>
      <td>2.9</td>
      <td><b>16</b></td>
      <td><b>253</b></td>
      <td>6.0</td>
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
      <td>64</td>
      <td><b>63</b></td>
      <td><b>29.8</b></td>
    </tr>
    <tr>
      <th>natasha</th>
      <td><b>2.0</b></td>
      <td><b>1</b></td>
      <td><b>160</b></td>
      <td>8.8</td>
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

## Support

- Chat — https://telegram.me/natural_language_processing
- Issues — https://github.com/natasha/slovnet/issues

## Development

Rent GPU:

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

yc compute instance list
yc compute instance delete fhmj2ftcm32qgqt4igjf

```

Setup instance:

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

Intall dev:

```bash
pip3 install -r slovnet/requirements/dev.txt
pip3 install -e slovnet

```
