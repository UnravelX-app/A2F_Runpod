FROM nvcr.io/nim/nvidia/audio2face-3d:1.3.16

USER root

WORKDIR /app

COPY ../a2f-wrapper/requirements.txt .
RUN pip install --index-url https://pypi.org/simple/ -r requirements.txt

COPY ../a2f-wrapper/app ./app

ENV A2F_MODE=local
ENV A2F_GRPC_ADDR=localhost:52000
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", \
  "/opt/nim/start-server.sh & \
  until grpc_health_probe -addr=localhost:52000 2>/dev/null || \
        python -c \"import grpc; ch=grpc.insecure_channel('localhost:52000'); grpc.channel_ready_future(ch).result(timeout=3)\" 2>/dev/null; do \
    echo 'Waiting for A2F gRPC...'; sleep 5; \
  done && \
  uvicorn app.main:app --host 0.0.0.0 --port 8080"]
