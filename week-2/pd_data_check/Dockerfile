FROM python:3.8

WORKDIR /pandas_profiling
COPY measure.py /pandas_profiling
COPY requirements.txt /pandas_profiling

RUN pip install -r requirements.txt

CMD python measure.py
