# syntax=docker/dockerfile:1

FROM python:3.11

# Install necessary tools and the ODBC Driver for SQL Server
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    curl \
    apt-transport-https \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

#Copy the current directory contents into the container at /app
COPY ./ /app/
COPY /app /app

EXPOSE 3100

CMD ["gunicorn", "main:app"]

RUN rm -f /bin/sh /bin/bash