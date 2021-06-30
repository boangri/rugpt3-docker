import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, render_template
import json
import os


np.random.seed(42)
torch.manual_seed(42)
torch.__version__

path = "sberbank-ai/rugpt3large_based_on_gpt2"
tok = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path) # .cuda()

#prompt = "Сингапур стал первой страной, разрешившей"

do_sample=True
max_length=30
repetition_penalty=5.0
top_k=5
top_p=0.95
temperature=1
num_beams=10
no_repeat_ngram_size=3

#input_ids = tok.encode(prompt, return_tensors="pt")
#out = model.generate(
#      input_ids,
#      max_length=max_length,
#      repetition_penalty=repetition_penalty,
#      do_sample=do_sample,
#      top_k=top_k, top_p=top_p, temperature=temperature,
#      num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
#      )
#print(len(out[0]))
#generated = list(map(tok.decode, out))
#print(generated[0])


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html") # темплейты берутся из каталога templates

# здесь будут обрабатываться AJAX-запросы на распознавание цифры
@app.route('/predict/', methods=['POST'])
def predict():
    json_str = request.get_data()
    data = json.loads(json_str)
    prompt = data['prompt']
    input_ids = tok.encode(prompt, return_tensors="pt")
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )
    generated = list(map(tok.decode, out))
    answer = {"answer": generated[0]}
    response = json.dumps(answer)
    return response

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

