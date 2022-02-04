FROM python:3.9

COPY requirements.txt .
COPY . /home/Document_Similarity_Quora_project

RUN pip install -r requirements.txt

WORKDIR /home/Document_Similarity_Quora_project

CMD ["bash"]