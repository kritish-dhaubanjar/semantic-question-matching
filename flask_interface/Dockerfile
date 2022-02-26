FROM python:3.6

WORKDIR /opt
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV FLASK_APP=app.py

CMD [ "python", "-m", "flask", "run", "--without-threads", "--host=0.0.0.0" ]

