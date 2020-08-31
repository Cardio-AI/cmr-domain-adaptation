FROM python:3.6
COPY ./src/app /code
COPY requirements.txt /temp/
WORKDIR /code
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /temp/requirements.txt
CMD ["python", "app.py"]