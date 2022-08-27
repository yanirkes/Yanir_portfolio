FROM python:3.9

WORKDIR /tree-app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src ./src

CMD ["python", "./src/RestAPI.py"]