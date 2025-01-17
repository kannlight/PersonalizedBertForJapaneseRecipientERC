import random
import json
import os

batch_size = 8
dataset_dir = ''

def generate_batch_user(filename):
    # [a,b,c,d,e,f] -> [[a,b],[c,d],[e,f]] (ex.batch size 2)
    talk_data = []
    data = {}
    with open(dataset_dir/+filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 対話を追加
    if 'data' in data:
        for talk in data['data']:
            talk_data.append(talk)
    
    # 対話をシャッフル
    random.shuffle(talk_data)
    # バッチサイズに従って分割
    batch_list = []
    for i in range(len(talk_data) // batch_size):
        batch_list.append(talk_data[i:i+batch_size])

    return batch_list

def main():
    # 各ユーザーのファイルからミニバッチ(のようなリスト)を作成
    batches = []
    for filename in os.listdir(dataset_dir):
        batch_list = generate_batch_user(filename)
        batches = batches + batch_list
    
    # ミニバッチをシャッフル
    random.shuffle(batches)
    # 6:2:2で学習:検証:テストデータに分割
    n = len(batches)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = {"data": batches[:n_train]}
    dataset_val = {"data": batches[n_train:n_train+n_val]}
    dataset_test = {"data": batches[n_train+n_val:]}
    # 保存
    with open('DatasetTrain.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_train, f, indent=4, ensure_ascii=False)
    with open('DatasetVal.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_val, f, indent=4, ensure_ascii=False)
    with open('DatasetTest.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_test, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()