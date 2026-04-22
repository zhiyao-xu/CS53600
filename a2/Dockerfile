FROM ubuntu:24.04

# Suppress apt interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Copy source files
COPY client.py run_experiment.py plot.py ml_model.py ./

# Create output directories
RUN mkdir -p data results

# Note: TCP_INFO and congestion control behavior depend on the HOST Linux kernel.
# Run with --network=host for realistic TCP measurements.
# Example:
#   docker build -t a2 .
#   docker run --network=host a2 --num-servers 10 --duration 60

ENTRYPOINT ["python3", "run_experiment.py"]
CMD ["--num-servers", "10", "--duration", "60", "--interval", "1.0"]
