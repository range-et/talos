# Talos

Talos is a Flask-based API that uses DistilBERT and MPT-7B for text generation and embedding. It's designed to run on various platforms and can utilize both CPU and GPU resources. This project aims to provide a lightweight solution for low-intensity data enrichment tasks.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Docker Deployment](#docker-deployment)
7. [Performance Considerations](#performance-considerations)

## Features

- Text generation (question answering) using DistilBERT and MPT-7B
- Text embedding generation
- Batch processing for tokenization, embedding, and text generation
- Flask-based RESTful API
- Docker support for deployment
- Automatic GPU utilization when available (CUDA for NVIDIA, MPS for Apple Silicon)

## Prerequisites

- Python 3.9+
- pip and pipenv
- Docker (for containerized deployment)
- CUDA-compatible GPU (optional, for GPU acceleration on systems with NVIDIA GPUs)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/talos.git
   cd talos
   ```

2. Set up the Python environment:
   ```
   pipenv install
   pipenv shell
   ```

3. Download the models:
    ```
    bash download_models.sh
    ```

4. Run the application:
   ```
   python app.py
   ```

The server will start on `http://localhost:3000`.

## API Endpoints

### 1. Text Generation (Question Answering)

- **URL:** `/generate`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "context": "string",
  "question": "string",
  "model": "string" // "distilbert" or "mpt"
}
```

**Response Schema:**
```json
{
  "answer": "string"
}
```

### 2. Embedding Generation

- **URL:** `/embed`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "text": "string",
  "model": "string" // "distilbert" or "mpt"
}
```

**Response Schema:**
```json
{
  "embeddings": [float]
}
```

(Other endpoints remain the same as in the original README)

## Usage Examples

### Text Generation

```bash
# DistilBERT
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox do?",
    "model": "distilbert"
  }'

# MPT-7B
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox do?",
    "model": "mpt"
  }'
```

### Embedding Generation

```bash
# DistilBERT
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "model": "distilbert"
  }'

# MPT-7B
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "model": "mpt"
  }'
```

(Other examples remain the same, just add the "model" parameter)

## Model Comparison and When to Use Each

### DistilBERT
- Context window: Approximately 512 tokens (roughly 300-400 words)
- Suitable for: Quick tasks, short texts, scenarios requiring faster processing
- Use when: Speed is prioritized over accuracy, or when dealing with shorter texts

### MPT-7B
- Context window: Approximately 2048 tokens (roughly 1500-2000 words)
- Suitable for: Longer contexts, more complex tasks
- Use when: Better performance is needed on longer texts or more nuanced tasks

For texts longer than a few paragraphs, MPT-7B may be more appropriate. For shorter snippets or when speed is crucial, DistilBERT is often sufficient.

## Docker Deployment

1. Build the Docker image:
   ```
   docker build -t talos-api .
   ```

2. Run the container:
   ```
   docker run -p 5000:3000 talos-api
   ```

For GPU support on Linux with NVIDIA GPUs:
```
docker run --gpus all -p 3000:3000 talos-api
```

## Performance Considerations

- The API automatically selects the best available hardware.
- Initial requests may be slower due to model loading.
- For production use, consider using a production-grade WSGI server like Gunicorn (configured in the Dockerfile).
- Consider implementing request queuing and load balancing for high-traffic scenarios.
- Batch processing endpoints can improve efficiency for multiple inputs.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.