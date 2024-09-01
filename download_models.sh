#!/bin/bash

# Directory to store the models
MODEL_DIR="./models"

# Function to download the MPT-7B model
download_mpt7b() {
    echo "Downloading MPT-7B model..."
    mkdir -p "$MODEL_DIR/mpt-7b"
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b'); model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-7b'); tokenizer.save_pretrained('$MODEL_DIR/mpt-7b'); model.save_pretrained('$MODEL_DIR/mpt-7b')"
    echo "MPT-7B model downloaded successfully."
}

# Check if MPT-7B model exists
if [ ! -d "$MODEL_DIR/mpt-7b" ]; then
    echo "MPT-7B model not found."
    download_mpt7b
else
    echo "MPT-7B model found in $MODEL_DIR/mpt-7b"
fi

# Check if DistilBERT model exists
if [ ! -d "$MODEL_DIR/distilbert" ]; then
    echo "DistilBERT model not found. Downloading..."
    mkdir -p "$MODEL_DIR/distilbert"
    python -c "from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertModel; tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad'); embedding_model = DistilBertModel.from_pretrained('distilbert-base-uncased'); tokenizer.save_pretrained('$MODEL_DIR/distilbert'); qa_model.save_pretrained('$MODEL_DIR/distilbert'); embedding_model.save_pretrained('$MODEL_DIR/distilbert')"
    echo "DistilBERT model downloaded successfully."
else
    echo "DistilBERT model found in $MODEL_DIR/distilbert"
fi

echo "All required models are now available."