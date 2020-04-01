FROM python:3.6

RUN S3=https://storage.yandexcloud.net/natasha-slovnet \
    && wget $S3/01_bert_news/rubert/vocab.txt \
    && wget $S3/04_bert_syntax/model/rels_vocab.txt \
    && wget $S3/01_bert_news/model/emb.pt \
    && wget $S3/04_bert_syntax/model/encoder.pt \
    && wget $S3/04_bert_syntax/model/head.pt \
    && wget $S3/04_bert_syntax/model/rel.pt

COPY requirements/app.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-syntax-bert/app.py