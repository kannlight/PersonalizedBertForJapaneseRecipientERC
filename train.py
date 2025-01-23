import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertPreTrainedModel, BertModel, get_scheduler
from transformers.modeling_outputs import SequenceClassifierOutput
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import math

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

pack_size = 8

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

ICFweight = 1 / cf
ICFweight = ICFweight / torch.sum(ICFweight)

def tokenize_pack(filename):
    # データセットの対話パック(対話データの配列)をトークン化して返す関数
    dataset_for_loader = []
    # データセットから対話データを読み込む
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各対話をトークン化して追加
    if 'data' in data:
        for pack in data['data']:
            text1 = []
            text2 = []
            for talk in pack:
                # 1発話目(受信者の発話)と2発話目(送信者の発話)を取り出す（複数続いたら改行で繋げる）
                t1 = []
                t2 = []
                for utter in talk['talk']:
                    if utter['type'] == 1:
                        t1.append(utter['utter'])
                    if utter['type'] == 2:
                        t2.append(utter['utter'])
                text1.append('\n'.join(t1))
                text2.append('\n'.join(t2))
            # 1パック分の対話データをまとめてトークン化
            token=tokenizer(
                text1, text2,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            )
            # ラベル付け
            labels = []
            for talk in pack:
                labels.append(talk['label'])
            # バリューをテンソル化して追加
            token['labels'] = torch.tensor(labels)
            dataset_for_loader.append(token)
    return dataset_for_loader

class PersonalizerAttention(nn.Module):
    # パック間の特徴量の内積をとるSelf-Attention
    # 入力[CLS](batch_size,pack_size,hidden_size=768)、出力[personalized_CLS](batch_size,pack_size,hidden_size=768)
    def __init__(self, config):
        super().__init__()

        # Multi-head Attentionにするための設定
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        """
        hidden_size = 768
        num_attention_heads = 12
        attention_head_size = 64
        all_head_size = 768
        """
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q,K,Vを用意するための全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        # Multi-head Attention用にテンソルを変形(batch_size,pack_size,hidden_size -> batch_size,num_attention_heads,pack_size,attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        # Q,K,Vを用意
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Q * K^t
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # QKt / √d
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Softmax( QKt/√d )
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # ドロップアウト
        attention_probs = self.dropout(attention_probs)

        # Softmax(QKt/√d) * V
        context_layer = torch.matmul(attention_probs, value_layer)

        # Multi-head用に変形していたテンソルを元に戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class PersonalizedBertForSequenceClassification(BertPreTrainedModel):
    # 同一人物に関する文章を受け取って個別化した分類タスクを行うモデル
    # 入力(batch_size,pack_size,seq_len)、出力(batch_size,pack_size,num_labels)
    def __init__(self, config):
        super().__init__(config)
        # 分類クラス数
        self.num_labels = config.num_labels
        # 特徴量の次元
        self.hidden_size = config.hidden_size

        self.config = config

        # bert
        self.bert = BertModel(config)
        # LayerNorm
        self.LayerNorm = nn.LayerNorm(self.hidden_size, config.layer_norm_eps)
        # Attention層前のドロップアウト
        personalizer_dropout = 0.1
        self.dropout1 = nn.Dropout(personalizer_dropout)
        # 専用のAttention層
        self.attention = PersonalizerAttention(config)
        # 全結合層前のドロップアウト
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout2 = nn.Dropout(classifier_dropout)
        # 全結合層
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        # 重みの初期化などをするPreTrainedModelのメソッド
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        input_ids,attention_mask,token_type_ids
        (batch,pack,512)
        labels
        (batch,pack,1)
        """
        # bertに並列計算させるため形式を変更
        # (batch,pack,512)->(batch*pack,512)
        shaped_ids = input_ids.view(-1,MAX_LENGTH)
        shaped_mask = attention_mask.view(-1,MAX_LENGTH)
        shaped_type = token_type_ids.view(-1,MAX_LENGTH)

        # bertの出力を得る
        # (batch*pack,512)->(batch*pack,512,768)
        bert_outputs = self.bert(
            input_ids=shaped_ids,
            attention_mask=shaped_mask,
            token_type_ids=shaped_type
        )
        # [CLS]を取得して、元の形状に戻す
        # (batch*pack,512,768)->(batch*pack,768)
        pooled_output = bert_outputs[1]
        # (batch*pack,768)->(batch,pack,768)
        pooled_output = pooled_output.view(-1,pack_size,self.hidden_size)
        # LayerNorm(Self-Attentionに入力する前に正規化すべき)とDropoutを適用
        # (batch,pack,768)->(batch,pack,768)
        pooled_output = self.LayerNorm(pooled_output)
        pooled_output = self.dropout1(pooled_output)

        # Self-Attentionによってバッチ間で相互作用
        # (batch,pack,768)->(batch,pack,768)
        personalized_output = self.attention(pooled_output)

        # ドロップアウトを適用して全結合層へ
        # (batch,pack,768)->(batch,pack,768)
        personalized_output = self.dropout2(personalized_output)
        # (batch,pack,768)->(batch,pack,num_labels)
        logits = self.classifier(personalized_output)

        # クロスエントロピーロスを計算して返す(batch*pack分ある損失の平均)
        # 形式を変更して計算(batch,pack,num_labels)->(batch*pack,num_labels)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits
            )
        else:
            return SequenceClassifierOutput(
                logits=logits
            )

class PersonalizedBertForJapaneseRecepientERC(pl.LightningModule):
    # 同一の受信者対話データ(2発話)を受け取って個別化した感情分類タスク(受信者感情)を行うモデル
    # 入力(batch_size,pack_size,seq_len)、出力(batch_size,pack_size,num_labels)
    def __init__(
            self,
            model_name=MODEL_NAME,
            num_labels=8,
            lr=0.001,
            weight_decay=0.01,
            dropout=None,
            warmup_steps=None,
            total_steps=None
        ):
        # model_name: 事前学習モデル
        # num_labels: ラベル数
        # pack_size: まとめて入力する対話の数
        # lr: 学習率(特に指定なければAdamのデフォルト値を設定)
        # weight_decay: 重み減衰の強度(L2正則化のような役割)
        # dropout: 全結合層の前に適用するdropout率、デフォルトでNoneとしているがこの場合はconfig.hidden_dropout_probが適用される(0.1など)
        # warmup_steps: ウォームアップの適用のステップ数
        # total_steps: 学習全体のステップ数
        super().__init__()
        self.save_hyperparameters()

        # 事前学習モデルのロード
        self.model = PersonalizedBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, classifier_dropout=dropout)

    # 学習データを受け取って損失を返すメソッド
    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )
        # 損失関数にCEではなくICFを用いる
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        loss = loss_func(output.logits.view(-1,self.hparams.num_labels), batch['labels'].view(-1,self.hparams.num_labels))
        self.log('train_loss', loss)
        return loss
    
    # 検証データを受け取って損失を返すメソッド
    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        val_loss = loss_func(output.logits.view(-1,self.hparams.num_labels), batch['labels'].view(-1,self.hparams.num_labels))
        self.log('val_loss', val_loss)

    def on_train_batch_start(self, batch, batch_idx):
        # 学習率の変化を記録
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)

    def configure_optimizers(self):
        # オプティマイザーの指定
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # ウォームアップを適用
        if self.hparams.warmup_steps is not None and self.hparams.total_steps is not None:
            scheduler = get_scheduler(
                name='linear',
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        else:
            return optimizer

def main():
    # データセットから対話パックをトークン化
    dataset_train = tokenize_pack('./DatasetForExperiment2/DatasetTrain.json')
    dataset_val = tokenize_pack('./DatasetForExperiment2/DatasetVal.json')
    # データローダ作成
    dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, num_workers=2, batch_size=4)

    # ハイパーパラメータ
    max_epochs = 10 # 学習のエポック数
    total_steps = len(dataloader_train) * max_epochs
    warmup_steps = int(0.1 * total_steps) # ウォームアップの適用期間
    lr = 3e-5 # 初期学習率
    wd = 0.1 # 重み減衰率
    dropout = 0.1 # 全結合前のドロップアウト率

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
        max_epochs = max_epochs, # 学習のエポック数
        callbacks = [checkpoint]
    )
    # ハイパーパラメータを指定してモデルをロード
    model = PersonalizedBertForJapaneseRecepientERC(
        lr=lr,
        weight_decay=wd,
        dropout=dropout,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    # ファインチューニング
    trainer.fit(model, dataloader_train, dataloader_val)

    # 結果表示
    print('best_model_path:', checkpoint.best_model_path)
    print('val_loss for best_model:', checkpoint.best_model_score)

    best_model = PersonalizedBertForJapaneseRecepientERC.load_from_checkpoint(
        checkpoint.best_model_path
    )
    best_model.model.save_pretrained('./model_transformers')

if __name__ == "__main__":
    main()