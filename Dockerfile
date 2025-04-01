FROM python:3.12

COPY . deep_orderbook

WORKDIR /deep_orderbook

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e . 

CMD [ "python", "-m", "deep_orderbook.consumers.recorder" ]
