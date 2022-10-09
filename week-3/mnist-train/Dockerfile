FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y make

RUN pip install -U pip
RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt /app

RUN pip install -r requirements.txt

COPY ./conf /app/conf
COPY ./mnist_train /app/mnist_train
COPY ./tests /app/tests
COPY ./Makefile /app/Makefile

ENV PYTHONPATH /app
ENV WANDB_API_KEY e1609af0abbe597f55eb7e22aa1cc8ff34acc5f8

CMD [ 'bash' ]
