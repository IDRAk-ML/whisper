FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /workspace

# Install Python + system packages (add libsndfile1)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-venv git curl libsndfile1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

COPY . .

RUN pip install -r requirements.txt

# Preload DeepFilterNet model to avoid runtime zip errors
RUN python3 -c "from df.enhance import init_df; init_df()"

# Pre-clone silero-vad to torch hub location
RUN mkdir -p /root/.cache/torch/hub && \
    git clone https://github.com/snakers4/silero-vad.git /root/.cache/torch/hub/snakers4_silero-vad_master

RUN mkdir -p /workspace/temp

EXPOSE 9005

# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9005", "--workers", "6"]
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 9005 --workers ${UVICORN_WORKERS:-4}"]
