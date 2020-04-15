FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN S3=https://storage.yandexcloud.net/natasha-slovnet \
    GH=https://github.com/natasha/navec/releases/download/v0.0.0 \
    && curl -O $S3/05_ner/model/shape.pt \
    && curl -O $S3/05_ner/model/encoder.pt \
    && curl -O $S3/05_ner/model/ner.pt \
    && curl -L $GH/navec_news_v1_1B_250K_300d_100q.tar > navec.tar

COPY requirements/app.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-ner/torch/app.py
