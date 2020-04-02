FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN S3=https://storage.yandexcloud.net/natasha-slovnet \
    && curl -O $S3/01_bert_news/rubert/vocab.txt \
    && curl -O $S3/01_bert_news/model/emb.pt \
    && curl -O $S3/02_bert_ner/model/encoder.pt \
    && curl -O $S3/02_bert_ner/model/ner.pt

COPY requirements/app.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-ner-bert/app.py
