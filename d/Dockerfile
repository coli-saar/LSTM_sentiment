FROM pytorch-0.3.1_cuda-9.0

RUN sudo apt-get update
RUN sudo apt-get install -y gcc

RUN pip install comet_ml visdom torchwordemb

WORKDIR /
COPY glove.6B.50d.txt.gz /glove.6B.50d.txt.gz
WORKDIR /glove
RUN sudo gunzip /glove.6B.50d.txt.gz

WORKDIR /logs
WORKDIR /code/checkpoints


WORKDIR /

