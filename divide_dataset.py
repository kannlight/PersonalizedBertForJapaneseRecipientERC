import random
import json
import os

pack_size = 8
dataset_dir = 'DatasetByLuke'

def make_package(filename):
    # パックサイズに従って対話データを分割
    # [a,b,c,d,e,f] -> [[a,b],[c,d],[e,f]] (ex.batch size 2)
    talk_data = []
    data = {}
    with open(dataset_dir+'/'+filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 対話を追加
    if 'data' in data:
        for talk in data['data']:
            talk_data.append(talk)
    
    # 対話をシャッフル
    random.shuffle(talk_data)
    # パックサイズに従って分割
    packs = []
    for i in range(len(talk_data) // pack_size):
        packs.append(talk_data[i:i+pack_size])

    return packs

def main():
    # 各ユーザーのファイルからパックを作成
    packs = []
    for filename in os.listdir(dataset_dir):
        pack = make_package(filename)
        packs = packs + pack
    
    # パックをシャッフル
    random.shuffle(packs)
    # 6:2:2で学習:検証:テストデータに分割
    n = len(packs)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = {"data": packs[:n_train]}
    dataset_val = {"data": packs[n_train:n_train+n_val]}
    dataset_test = {"data": packs[n_train+n_val:]}
    # 保存
    with open('DatasetTrain.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_train, f, indent=4, ensure_ascii=False)
    with open('DatasetVal.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_val, f, indent=4, ensure_ascii=False)
    with open('DatasetTest.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_test, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()