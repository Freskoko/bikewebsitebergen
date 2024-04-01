FROM python:3.10-slim-bullseye

COPY --chown=root:root ./pyproject.toml ./pyproject.toml

RUN pip install --no-cache-dir poetry

WORKDIR /

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# poetry
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

RUN useradd --create-home appuser
USER appuser

WORKDIR /src

COPY --chown=appuser . .

EXPOSE 5000

ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask","run"]

# docker build . -t apitest
# docker run -d -p 5000:5000 --name apitestrun apitest
