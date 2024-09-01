import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MPTModel:
    def __init__(self):
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        model_path = "./models/mpt-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token to the eos_token if pad_token is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def generate(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the last layer of the model output (logits)
        embeddings = outputs.logits.mean(dim=1).squeeze().cpu().tolist()
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
        
        return response

    def batch_embed(self, texts, max_length, pooling):
        inputs = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the last layer of the model output (logits)
        logits = outputs.logits
        
        if pooling == 'mean':
            embeddings = logits.mean(dim=1)
        elif pooling == 'max':
            embeddings = logits.max(dim=1).values
        elif pooling == 'cls':
            embeddings = logits[:, 0, :]
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
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_length, padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
            
            result = {
                'question': question,
                'context': context,
                'answer': answer,
            }
            results.append(result)
        
        return results