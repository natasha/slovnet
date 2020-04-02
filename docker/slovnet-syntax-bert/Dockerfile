FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN S3=https://storage.yandexcloud.net/natasha-slovnet \
    && curl -O $S3/01_bert_news/rubert/vocab.txt \
    && curl -O $S3/04_bert_syntax/model/rels_vocab.txt \
    && curl -O $S3/01_bert_news/model/emb.pt \
    && curl -O $S3/04_bert_syntax/model/encoder.pt \
    && curl -O $S3/04_bert_syntax/model/head.pt \
    && curl -O $S3/04_bert_syntax/model/rel.pt

COPY requirements/app.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-syntax-bert/app.py