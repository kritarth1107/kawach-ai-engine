FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg age \
 && install -d /usr/share/postgresql-common/pgdg \
 && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
 && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
 && curl -fsSL https://pgp.mongodb.com/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-7.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/mongodb-7.gpg] https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" > /etc/apt/sources.list.d/mongodb.list \
 && apt-get update && apt-get install -y --no-install-recommends postgresql-client-16 mongodb-database-tools \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3==1.35.* pymongo==4.10.*

WORKDIR /backup
COPY backup.py .
ENTRYPOINT ["python", "backup.py"]
CMD ["run"]
