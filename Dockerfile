FROM tensorflow/tensorflow:latest-gpu

#set up environment
RUN apt-get update --fix-missing
RUN apt-get install -y apt-utils curl python3.8 python3-pip python3.8-dev git unzip
RUN apt-get upgrade -y
RUN python3.8 -m pip install --upgrade pip

COPY src /src

RUN python3.8 -m pip install -f https://download.pytorch.org/whl/torch_stable.html -r /src/requirements.txt

CMD ["python3.8", "/src/calculate_document_similarity.py"]
