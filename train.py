import random
from tqdm import tqdm
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import classification_report

MODEL_NAME = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
tokenizer =BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

CATEGORIES = [
    'joy',
    'sadness',
    'anticipation',
    'surprise',
    'anger',
    'fear',
    'disgust',
    'trust'
]
MAX_LENGTH = 512

class_frequency = [
    4450.92560337,
     1259.4576316,
     2719.97370069,
     1071.27600888,
     96.64192807,
     686.30217162,
     237.07288376,
     12.35007948
]

cf = torch.tensor(class_frequency).cuda()

ICFweight = torch.sum(cf) / cf # 割合の逆数

def tokenize_data(filename):
    # データセットの対話データをトークン化して返す関数
    dataset_for_loader = []
    # データセットから対話データを読み込む
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各対話をトークン化して追加
    if 'data' in data:
        for talk in data['data']:
            # 1発話目(受信者の発話)と2発話目(送信者の発話)を取り出す（複数続いたら改行で繋げる）
            t1 = []
            t2 = []
            for utter in talk['talk']:
                if utter['type'] == 1:
                    t1.append(utter['utter'])
                if utter['type'] == 2:
                    t2.append(utter['utter'])
            text1 = '\n'.join(t1)
            text2 = '\n'.join(t2)
            # トークン化
            token=tokenizer(
                text1, text2,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )
            # ラベル付け
            token['labels'] = talk['label']
            # バリューをテンソル化して追加
            token = { k: torch.tensor(v) for k, v in token.items() }
            dataset_for_loader.append(token)
    return dataset_for_loader

class BertForJapaneseRecepientERC(pl.LightningModule):
    def __init__(self, model_name=MODEL_NAME, num_labels=8, lr=0.001, weight_decay=0.01, dropout=None):
        # model_name: 事前学習モデル
        # num_labels: ラベル数
        # lr: 学習率(特に指定なければAdamのデフォルト値を設定)
        # weight_decay: 重み減衰の強度(L2正則化のような役割)
        super().__init__()
        self.save_hyperparameters()

        # 事前学習モデルのロード
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, classifier_dropout=dropout)

    # 学習データを受け取って損失を返すメソッド
    def training_step(self, batch):
        output = self.model(**batch)
        # 損失関数にCEではなくICFを用いる
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        loss = loss_func(output.logits, batch['labels'])
        self.log('train_loss', loss)
        return loss
    
    # 検証データを受け取って損失を返すメソッド
    def validation_step(self, batch):
        output = self.model(**batch)
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        val_loss = loss_func(output.logits, batch['labels'])
        self.log('val_loss', val_loss)

    # # テストデータを受け取って評価指標を計算
    # def test_step(self, batch):
    #     # テストデータのラベル
    #     true_labels = batch.pop('labels')
    #     # モデルが出力した分類スコアから、最大値となるクラスを取得
    #     output =self.model(**batch)
    #     predicted_labels = output.logits.argmax(-1)
    #     # precision, recall, f1, データ数 をクラス毎、ミクロ、マクロ、加重平均で算出
    #     self.log('test_report', classification_report(true_labels, predicted_labels, target_names=CATEGORIES))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

def main():
    # データセットから対話データをトークン化
    dataset_train = tokenize_data('DatasetTrain.json')
    dataset_val = tokenize_data('DatasetVal.json')
    # データローダ作成
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=256)

    # ファインチューニングの設定
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'min', # monitorの値が小さいモデルを保存
        save_top_k = 1, # 最小のモデルだけ保存
        save_weights_only = True, # モデルの重みのみを保存
        dirpath='model/' # 保存先
    )
    # 学習方法の指定
    trainer = pl.Trainer(
        accelerator = 'gpu', # 学習にgpuを使用
        devices = 1, # gpuの個数
        max_epochs = 10, # 学習のエポック数
        callbacks = [checkpoint]
    )
    # 学習率を指定してモデルをロード
    model = BertForJapaneseRecepientERC(lr=3e-5, weight_decay=0, dropout=0.2)
    # ファインチューニング
    trainer.fit(model, dataloader_train, dataloader_val)

    # 結果表示
    print('best_model_path:', checkpoint.best_model_path)
    print('val_loss for best_model:', checkpoint.best_model_score)

    # テストデータで評価
    # test = trainer.test(dataloaders=dataloader_test)
    # print(test[0]['test_report'])

    best_model = BertForJapaneseRecepientERC.load_from_checkpoint(
        checkpoint.best_model_path
    )
    best_model.model.save_pretrained('./model_transformers')

if __name__ == "__main__":
    main()