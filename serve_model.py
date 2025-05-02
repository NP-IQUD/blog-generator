from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Force CPU mode
device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained("./tinyllama-blog-finetuned").to(device)
tokenizer = AutoTokenizer.from_pretrained("./tinyllama-blog-finetuned")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.9, temperature=0.8)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({ 'content': text })

if __name__ == '__main__':
    app.run(port=8000)
