from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertModel
import torch

app = Flask(__name__)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

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

if __name__ == '__main__':
    app.run(debug=True)