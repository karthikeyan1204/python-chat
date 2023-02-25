import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
model = AutoModelForCausalLM.from_pretrained('gpt2-large')

# Set the model to evaluation mode
model.eval()

@app.route('/generate-text', methods=['POST'])
def generate_text():
    # Get the prompt from the request body
    prompt = request.json.get('prompt')

    # Encode the prompt as input for the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text based on the input prompt
    output = model.generate(
        input_ids=input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode the generated text and return it as a JSON response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt_len = len(prompt)
    generated_text = generated_text[prompt_len:]
    response = {'generated_text': generated_text}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
