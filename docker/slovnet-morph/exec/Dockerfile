FROM python:3.6

RUN curl -O https://storage.yandexcloud.net/natasha-slovnet/06_morph/pack/slovnet_morph_news_v1.tar

RUN pip install aiohttp==3.6.1

COPY requirements/main.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD python docker/slovnet-morph/exec/app.py
