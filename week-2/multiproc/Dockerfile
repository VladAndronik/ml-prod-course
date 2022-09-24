FROM python:3.8

WORKDIR /multiprocess
COPY main.py /multiprocess
COPY utils.py /multiprocess
COPY requirements.txt /multiprocess

RUN pip install -r requirements.txt

CMD python main.py
