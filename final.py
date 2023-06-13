import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from flask import Flask, request
from test import BiLSTMLighting,BiLSTMClassifier
bert_model_path = 'bert-base-chinesebert-base-chinese/'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertModel.from_pretrained(bert_model_path)
from flask import render_template

app = Flask(__name__)

def text_to_tensor(text):
    inputs = token.batch_encode_plus(
        [text],
        max_length=200,  # 最大长度
        truncation=True,  # 截断长句子
        padding='max_length',  # 填充至最大长度
        return_tensors='pt'  # 返回 PyTorch 张量
    )

    get_input_ids = inputs['input_ids']
    get_attention_mask = inputs['attention_mask']
    get_token_type_ids = inputs['token_type_ids']

    get_input_ids=get_input_ids.to('cuda')
    get_attention_mask=get_attention_mask.to('cuda')
    get_token_type_ids=get_token_type_ids.to('cuda')

    return get_input_ids, get_attention_mask, get_token_type_ids



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=my_model_PATH, drop=dropout, hidden_dim=rnn_hidden,output_dim=class_num)
    text = request.form['text']
    input_ids, attention_mask, token_type_ids = text_to_tensor(text)
    output = model(input_ids=(input_ids), attention_mask=(attention_mask), token_type_ids=(token_type_ids))
    pred = int(torch.argmax(output,dim=1))
    if pred==0:
        prediction='消极'
    elif pred==1:
        prediction='中性'
    elif pred==2:
        prediction ='积极'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    batch_size = 128
    epochs = 30
    dropout = 0.5
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 3
    lr = 0.001
    # 指定测试使用的模型
    my_model_PATH = './lightning_logs/version_3/checkpoints/epoch=29-step=3840.ckpt'
    token = bert_tokenizer
    app.run(debug=True)