from flask import Flask, request, jsonify
from models import get_model

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    context = data.get('context', '')
    question = data.get('question', '')
    model_name = data.get('model', 'distilbert')

    model = get_model(model_name)
    answer = model.generate(context, question)

    return jsonify({'answer': answer})

@app.route('/embed', methods=['POST'])
def generate_embedding():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model', 'distilbert')

    model = get_model(model_name)
    embeddings = model.embed(text)

    return jsonify({'embeddings': embeddings})

@app.route('/batch-tokenize', methods=['POST'])
def batch_tokenize():
    data = request.json
    texts = data.get('texts', [])
    max_length = data.get('max_length', 512)
    padding = data.get('padding', 'max_length')
    truncation = data.get('truncation', True)
    model_name = data.get('model', 'distilbert')

    model = get_model(model_name)
    tokenized = model.batch_tokenize(texts, max_length, padding, truncation)

    return jsonify(tokenized)

@app.route('/batch-embed', methods=['POST'])
def batch_embed():
    data = request.json
    texts = data.get('texts', [])
    max_length = data.get('max_length', 512)
    pooling = data.get('pooling', 'mean')
    model_name = data.get('model', 'distilbert')

    model = get_model(model_name)
    embeddings = model.batch_embed(texts, max_length, pooling)

    return jsonify(embeddings)

@app.route('/batch-generate', methods=['POST'])
def batch_generate():
    data = request.json
    contexts = data.get('contexts', [])
    questions = data.get('questions', [])
    max_length = data.get('max_length', 512)
    model_name = data.get('model', 'distilbert')

    model = get_model(model_name)
    results = model.batch_generate(contexts, questions, max_length)

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)