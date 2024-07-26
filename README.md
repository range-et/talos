# Talos

This project implements a Flask-based API that uses DistilBERT for text generation (question answering) and embedding generation. It's designed to run efficiently on various platforms (Mac, Linux, Windows) and can utilize both CPU and GPU resources. This is good as a tiny LLM to perform low intensity data enrichment tasks. If you have recommendations for other models - please feel free to raise an issue.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Docker Deployment](#docker-deployment)
7. [Performance Considerations](#performance-considerations)

## Features

- Text generation (question answering) using DistilBERT
- Text embedding generation using DistilBERT
- Batch processing for tokenization, embedding, and text generation
- Flask-based RESTful API
- Docker support for easy deployment
- Automatic GPU utilization when available (CUDA for NVIDIA, MPS for Apple Silicon)

## Prerequisites

- Python 3.9+
- pip and pipenv
- Docker (for containerized deployment)
- CUDA-compatible GPU (optional, for GPU acceleration on systems with NVIDIA GPUs)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/distilbert-flask-api.git
   cd distilbert-flask-api
   ```

2. Set up the Python environment:
   ```
   pipenv install
   pipenv shell
   ```

3. Run the application:
   ```
   python app.py
   ```

The server will start on `http://localhost:5000`.

## API Endpoints

### 1. Text Generation (Question Answering)

- **URL:** `/generate`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "context": "string",
  "question": "string"
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
  "text": "string"
}
```

**Response Schema:**
```json
{
  "embeddings": [float]
}
```

### 3. Batch Tokenization

- **URL:** `/batch-tokenize-extended`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "texts": ["string"],
  "max_length": int,
  "padding": "string",
  "truncation": boolean
}
```

**Response Schema:**
```json
{
  "input_ids": [[int]],
  "attention_mask": [[int]],
  "token_type_ids": [[int]]
}
```

### 4. Batch Embedding

- **URL:** `/batch-embed-extended`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "texts": ["string"],
  "max_length": int,
  "pooling": "string"
}
```

**Response Schema:**
```json
{
  "embeddings": [[float]],
  "pooling_method": "string"
}
```

### 5. Batch Text Generation

- **URL:** `/batch-generate-extended`
- **Method:** `POST`
- **Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "contexts": ["string"],
  "questions": ["string"],
  "max_length": int
}
```

**Response Schema:**
```json
{
  "results": [
    {
      "question": "string",
      "context": "string",
      "answer": "string",
      "start_index": int,
      "end_index": int,
      "start_score": float,
      "end_score": float
    }
  ]
}
```

## Usage Examples

### Text Generation

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox do?"
  }'
```

Expected response:
```json
{
  "answer": "jumps over the lazy dog"
}
```

### Embedding Generation

```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog."
  }'
```

Expected response:
```json
{
  "embeddings": [-0.123, 0.456, -0.789, ..., 0.012]
}
```

### Batch Tokenization

```bash
curl -X POST http://localhost:5000/batch-tokenize-extended \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "How are you?"],
    "max_length": 128,
    "padding": "max_length",
    "truncation": true
  }'
```

### Batch Embedding

```bash
curl -X POST http://localhost:5000/batch-embed-extended \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "How are you?"],
    "max_length": 128,
    "pooling": "mean"
  }'
```

### Batch Text Generation

```bash
curl -X POST http://localhost:5000/batch-generate-extended \
  -H "Content-Type: application/json" \
  -d '{
    "contexts": ["The quick brown fox jumps over the lazy dog.", "Python is a programming language."],
    "questions": ["What does the fox do?", "What is Python?"],
    "max_length": 384
  }'
```

## Docker Deployment

1. Build the Docker image:
   ```
   docker build -t distilbert-api .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 distilbert-api
   ```

For GPU support on Linux with NVIDIA GPUs:
```
docker run --gpus all -p 5000:5000 distilbert-api
```

## Performance Considerations

- The API automatically selects the best available hardware:
  - On systems with NVIDIA GPUs, it will use CUDA.
  - On Apple Silicon Macs, it will use the Metal Performance Shaders (MPS) backend.
  - On systems without GPU support, it will fall back to CPU.
- For production use, consider using a production-grade WSGI server like Gunicorn (already configured in the Dockerfile).
- The first request might be slow as the model is loaded into memory. Subsequent requests will be faster.
- Consider implementing request queuing and load balancing for high-traffic scenarios.
- Batch processing endpoints can significantly improve efficiency for multiple inputs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.