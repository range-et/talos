import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertModel

class DistilBERTModel:
    def __init__(self):
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad').to(self.device)
        self.embedding_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)

    def generate(self, context, question):
        inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        return embeddings

    def batch_tokenize(self, texts, max_length, padding, truncation):
        tokenized = self.tokenizer(
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
        
        if 'token_type_ids' in tokenized and hasattr(tokenized['token_type_ids'], 'tolist'):
            response['token_type_ids'] = tokenized['token_type_ids'].tolist()
        
        return response

    def batch_embed(self, texts, max_length, pooling):
        inputs = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        if pooling == 'mean':
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif pooling == 'max':
            embeddings = outputs.last_hidden_state.max(dim=1).values
        elif pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError('Invalid pooling method')
        
        embeddings = embeddings.cpu().numpy().tolist()
        
        return {
            'embeddings': embeddings,
            'pooling_method': pooling
        }

    def batch_generate(self, contexts, questions, max_length):
        results = []
        
        for context, question in zip(contexts, questions):
            inputs = self.tokenizer.encode_plus(question, context, max_length=max_length, return_tensors='pt', truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            
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
        
        return results