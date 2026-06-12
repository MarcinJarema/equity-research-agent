# Obraz aplikacji FastAPI. Slim => mniejszy obraz, niższy koszt/szybszy pull.
FROM python:3.12-slim

# Nie buforuj bytecode'u i wypisuj logi od razu (lepsze logi w kontenerze).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Najpierw kopiujemy metadane zależności i instalujemy — ta warstwa cache'uje się,
# dopóki nie zmienisz pyproject.toml (szybsze przebudowy przy zmianach w kodzie).
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

EXPOSE 8000

# Produkcyjnie uruchamiamy bez --reload. Reload włączamy w docker-compose (dev).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
