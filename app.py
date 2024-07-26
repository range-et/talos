from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertModel
import torch
from typing import List, Dict, Any

app = Flask(__name__)

# Device selection (same as before)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load models and tokenizer (same as before)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad').to(device)
embedding_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Load models and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad').to(device)
embedding_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    context = data.get('context', '')
    question = data.get('question', '')

    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return jsonify({'answer': answer})

@app.route('/embed', methods=['POST'])
def generate_embedding():
    data = request.json
    text = data.get('text', '')

    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    
    # Use the mean of the last hidden state as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

    return jsonify({'embeddings': embeddings})

@app.route('/batch-tokenize-extended', methods=['POST'])
def batch_tokenize_extended():
    data = request.json
    texts: List[str] = data.get('texts', [])
    max_length: int = data.get('max_length', 512)
    padding: str = data.get('padding', 'max_length')
    truncation: bool = data.get('truncation', True)
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    tokenized = tokenizer(
        texts, 
        max_length=max_length, 
        padding=padding, 
        truncation=truncation, 
        return_tensors='pt'
    )
    
    response = {
        'input_ids': tokenized['input_ids'].tolist(),
        'attention_mask': tokenized['attention_mask'].tolist(),
    }
    
    # Only include token_type_ids if they are present and in the correct format
    if 'token_type_ids' in tokenized and hasattr(tokenized['token_type_ids'], 'tolist'):
        response['token_type_ids'] = tokenized['token_type_ids'].tolist()
    
    return jsonify(response)

@app.route('/batch-embed-extended', methods=['POST'])
def batch_embed_extended():
    data = request.json
    texts: List[str] = data.get('texts', [])
    max_length: int = data.get('max_length', 512)
    pooling: str = data.get('pooling', 'mean')
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    inputs = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    
    if pooling == 'mean':
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif pooling == 'max':
        embeddings = outputs.last_hidden_state.max(dim=1).values
    elif pooling == 'cls':
        embeddings = outputs.last_hidden_state[:, 0, :]
    else:
        return jsonify({'error': 'Invalid pooling method'}), 400
    
    embeddings = embeddings.cpu().numpy().tolist()
    
    return jsonify({
        'embeddings': embeddings,
        'pooling_method': pooling
    })

@app.route('/batch-generate-extended', methods=['POST'])
def batch_generate_extended():
    data = request.json
    contexts: List[str] = data.get('contexts', [])
    questions: List[str] = data.get('questions', [])
    max_length: int = data.get('max_length', 512)
    
    if not contexts or not questions or len(contexts) != len(questions):
        return jsonify({'error': 'Invalid input. Provide equal number of contexts and questions.'}), 400
    
    results = []
    
    for context, question in zip(contexts, questions):
        inputs = tokenizer.encode_plus(question, context, max_length=max_length, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
        result = {
            'question': question,
            'context': context,
            'answer': answer,
            'start_index': answer_start.item(),
            'end_index': answer_end.item(),
            'start_score': outputs.start_logits[0][answer_start].item(),
            'end_score': outputs.end_logits[0][answer_end-1].item()
        }
        results.append(result)
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)