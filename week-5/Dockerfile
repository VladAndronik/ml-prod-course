FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base

RUN pip install -U pip
RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt /app

RUN pip install -r requirements.txt

COPY ./serving /app/serving
COPY ./tests /app/tests

ENV PYTHONPATH /app

CMD [ 'bash' ]

FROM base as app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 serving/ui_app.py

FROM base as app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 2 serving.fast_api:app

FROM base as app-seldon

EXPOSE 5000

EXPOSE 9000

ENV MODEL_NAME SeldonAPI
ENV SERVICE_TYPE MODEL

COPY ./serving/seldon_api.py /app/SeldonAPI.py
COPY ./serving/__init__.py /app/serving/__init__.py
COPY ./serving/predictor.py /app/serving/predictor.py

RUN chown -R 8888 /app
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir /.config
RUN chmod 777 /.config

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
