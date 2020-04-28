FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN S3=https://storage.yandexcloud.net/ \
    && curl -O $S3/natasha-slovnet/07_syntax/model/shape.pt \
    && curl -O $S3/natasha-slovnet/07_syntax/model/encoder.pt \
    && curl -O $S3/natasha-slovnet/07_syntax/model/head.pt \
    && curl -O $S3/natasha-slovnet/07_syntax/model/rel.pt \
    && curl -O $S3/natasha-slovnet/07_syntax/model/rels_vocab.txt \
    && curl -L $S3/natasha-navec/navec_news_v1_1B_250K_300d_100q.tar > navec.tar

COPY requirements/app.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-syntax/torch/app.py
