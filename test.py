import torch
# from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy, recall, precision, f1_score  # lightning中的评估
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import torch.nn.functional as F

# 本地下好的bert-base-chinese模型
bert_model_path = 'bert-base-chinese/'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertModel.from_pretrained(bert_model_path)

# 洗掉label中的错误标签
data = pd.read_csv("./data/test_clean.csv")
label_col = 'label'
valid_labels = ['-1', '0', '1', -1, 0, 1]
label_mask = data[label_col].isin(valid_labels)
filtered_data = data[label_mask]
filtered_data.to_csv("./data/test_clean.csv", index=False)

data = pd.read_csv("./data/train_clean.csv")
label_col = 'label'
valid_labels = ['-1', '0', '1', -1, 0, 1]
label_mask = data[label_col].isin(valid_labels)
filtered_data = data[label_mask]
filtered_data.to_csv("./data/train_clean.csv", index=False)

data = pd.read_csv("./data/val_clean.csv")
label_col = 'label'
valid_labels = ['-1', '0', '1', -1, 0, 1]
label_mask = data[label_col].isin(valid_labels)
filtered_data = data[label_mask]
filtered_data.to_csv("./data/val_clean.csv", index=False)

# 加载csv文件
df = pd.read_csv('./data/test_clean.csv')
# 查看标签列中每个唯一值出现的次数
label_counts = df['label'].value_counts()
print(label_counts)


class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

    def __len__(self):
        return len(self.dataset)


# todo: 定义批处理函数
def collate_fn(data):
    """
    return:input_ids, attention_mask, token_type_ids, labels
    """
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = [int(label) for label in labels]
    labels = torch.LongTensor(labels)  # 该批次的labels
    return input_ids, attention_mask, token_type_ids, labels  # 得到四个tensor的元组


class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = bert_model
        # 去掉移至gpu
        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# todo
# 添加注意力机制
class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifierWithAttention, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = bert_model
        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        # 新增注意力层
        self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state
        out, (h_n, c_n) = self.lstm(embedded)

        # 注意力计算
        energy = self.attn(out)
        attention = F.softmax(energy, dim=1)
        weighted_out = torch.bmm(attention.permute(0, 2, 1), out).squeeze(1)

        output = self.fc(weighted_out)
        return output


class BiLSTMLightingwithatt(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        super().__init__()
        self.model = BiLSTMClassifierWithAttention(drop, hidden_dim, output_dim)  # 设置model
        self.criterion = nn.CrossEntropyLoss()  # 设置损失函数
        self.train_dataset = MydataSet('./data/train_clean.csv', 'train')
        self.val_dataset = MydataSet('./data/val_clean.csv', 'train')
        self.test_dataset = MydataSet('./data/test_clean.csv', 'train')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels + 1, num_classes=3)
        y = y.to(dtype=torch.float)

        # forward pass
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


# todo: 定义pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # 设置model
        self.criterion = nn.CrossEntropyLoss()  # 设置损失函数
        # self.train_dataset = MydataSet('./data/train_clean.csv', 'train')
        # self.val_dataset = MydataSet('./data/val_clean.csv', 'train')
        # self.test_dataset = MydataSet('./data/test_clean.csv', 'train')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    # def train_dataloader(self):
    #     train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
    #                               shuffle=True)
    #     return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels + 1, num_classes=3)
        # 将one_hot_labels类型转换成float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # 将[128, 1, 3]挤压为[128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 将loss输出在控制台
        return loss  # 必须把log返回回去才有用

    # def val_dataloader(self):
    #     val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    #     return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels + 1, num_classes=3)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    # def test_dataloader(self):
    #     test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    #     return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        target = labels + 1  # 用于待会儿计算acc和f1-score
        y = one_hot(target, num_classes=3)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == target).float().mean()

        loss = self.criterion(y_hat, y)
        self.log('loss', loss)
        # task: Literal["binary", "multiclass", "multilabel"],对应[二分类，多分类，多标签]
        #  average=None分别输出各个类别, 不加默认算平均
        re = recall(pred, target, task="multiclass", num_classes=class_num, average=None)
        pre = precision(pred, target, task="multiclass", num_classes=class_num, average=None)
        f1 = f1_score(pred, target, task="multiclass", num_classes=class_num, average=None)

        def log_score(name, scores):
            for i, score_class in enumerate(scores):
                self.log(f"{name}_class{i}", score_class)

        log_score("recall", re)
        log_score("precision", pre)
        log_score("f1", f1)
        self.log('acc', accuracy(pred, target, task="multiclass", num_classes=class_num))
        self.log('avg_recall', recall(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_precision', precision(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_f1', f1_score(pred, target, task="multiclass", num_classes=class_num, average="weighted"))


def test():
    # 加载之前训练好的最优模型参数
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=my_model_PATH, drop=dropout, hidden_dim=rnn_hidden,
                                                output_dim=class_num)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)


# 将文本转换成张量
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

    get_input_ids = get_input_ids.to('cuda')
    get_attention_mask = get_attention_mask.to('cuda')
    get_token_type_ids = get_token_type_ids.to('cuda')

    return get_input_ids, get_attention_mask, get_token_type_ids


def predict(text):
    # 加载模型权重
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=my_model_PATH, drop=dropout, hidden_dim=rnn_hidden,
                                                output_dim=class_num)
    # 将文本转换为张量
    input_ids, attention_mask, token_type_ids = text_to_tensor(text)
    # 使用模型进行推理
    output = model(input_ids=(input_ids), attention_mask=(attention_mask), token_type_ids=(token_type_ids))
    pred = torch.argmax(output, dim=1)
    if pred == 0:
        print('消极')
    elif pred == 1:
        print('中性')
    elif pred == 2:
        print('积极')


def collate_fn_predict(data):
    """
    return:input_ids, attention_mask, token_type_ids
    """
    sents = [i for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1

    input_ids = input_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    token_type_ids = token_type_ids.to('cuda')
    return input_ids, attention_mask, token_type_ids


def predict_batch(text):
    ans = {0: '消极', 1: '中性', 2: '积极'}
    # 加载模型权重
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=my_model_PATH, drop=dropout, hidden_dim=rnn_hidden,
                                                output_dim=class_num)
    loader = DataLoader(dataset=text, batch_size=batch_size, collate_fn=collate_fn_predict, shuffle=False)
    # 将文本转换为张量
    results = []
    for batch in loader:
        input_ids, attention_mask, token_type_ids = batch
        # 使用模型进行推理
        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pred = torch.argmax(output, dim=1)
        pred = pred.tolist()
        results.extend([ans[i] for i in pred])
    return results


def train():
    # 增加过拟合回调函数,提前停止,经过测试发现不太好用，因为可能会停止在局部最优值
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 监控对象为'val_loss'
        patience=4,  # 耐心观察4个epoch
        min_delta=0.0,  # 默认为0.0，指模型性能最小变化量
        verbose=True,  # 在输出中显示一些关于early stopping的信息，如为何停止等
    )
    # 增加回调最优模型，这个比较好用
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控对象为'val_loss'
        dirpath='./checkpoints/',  # 保存模型的路径
        filename='model-{epoch:02d}-{val_loss:.2f}',  # 最优模型的名称
        save_top_k=1,  # 只保存最好的那个
        mode='min'  # 当监控对象指标最小时
    )

    # Trainer可以帮助调试，比如快速运行、只使用一小部分数据进行测试、完整性检查等，
    # 详情请见官方文档https://lightning.ai/docs/pytorch/latest/debug/debugging_basic.html
    # auto自适应gpu数量
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=10, accelerator='gpu', devices="auto", fast_dev_run=False,
                      precision=16, callbacks=[checkpoint_callback])
    model = BiLSTMLightingwithatt(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer.fit(model)


if __name__ == '__main__':
    batch_size = 128
    epochs = 30
    dropout = 0.5
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 3
    lr = 0.001
    # 指定测试使用的模型
    my_model_PATH = './model/epoch=29-step=3840.ckpt'
    token = bert_tokenizer
    # train()
    # test()
    # print("请输入想说的话:")
    # test_text = input()
    # predict(test_text)
    text = list(pd.read_csv('processed.csv').content)
    result = predict_batch(text)
    with open('results.txt', "w", encoding="utf8") as f:
        f.writelines("\n".join(result))
