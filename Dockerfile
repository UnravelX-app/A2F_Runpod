FROM nvcr.io/nim/nvidia/audio2face-3d:1.3.16

USER root

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && pip install --no-cache-dir --index-url https://pypi.org/simple/ -r /app/requirements.txt

ENV PORT=8080

EXPOSE 8080 52000 8000

ENTRYPOINT ["/app/entrypoint.sh"]
