FROM python:3.10

ARG UID
ARG GID
ARG WORK_DIR=app

WORKDIR /$WORK_DIR

#RUN git config --global push.default simple

RUN groupadd --gid $GID myuser && useradd --no-create-home -u $UID --gid $GID myuser && chown -R myuser /app

RUN mkdir /home/myuser && chown -R myuser /home/myuser

#ENV PYTHONPATH=$PYTHONPATH:$(pwd)
ENV PYTHONPATH=/$WORK_DIR
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt
RUN pip --no-cache-dir install spacy
RUN python -m spacy download en_core_web_md

USER myuser

RUN git config --global user.email "276940@student.pwr.edu.pl"
RUN git config --global user.name "Repcak2000"
