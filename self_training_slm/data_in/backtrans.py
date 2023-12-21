import os
import csv
import tqdm
import argparse
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='sst2', help='sst2 or sst5')
args = parser.parse_args()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr",device=3)
pipe_bt = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-fr-en",device=3)
with open(os.path.join("data_in",args.task,"train_chat_vanilla.csv"), 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

ori_text = []
for row in (data):
    ori_text.append(row[1])

results = []
results_bt = []

for i in (range(len(ori_text)//32+1)):
    print(i)
    tmp_text = ori_text[i*32:(i+1)*32] if (i+1)*32 < len(ori_text) else ori_text[i*32:]
    batch_result = pipe(tmp_text,batch_size=32,max_length=768)
    mid_text = [tmp_result["translation_text"] for tmp_result in batch_result]
    results += (mid_text)

    
for i in (range(len(results)//32+1)):
    print(i)
    tmp_text = results[i*32:(i+1)*32] if (i+1)*32 < len(results) else results[i*32:]
    batch_result = pipe_bt(tmp_text,batch_size=32,max_length=768)
    bt_text = [tmp_result["translation_text"] for tmp_result in batch_result]
    results_bt += (bt_text)


for i in range(len(data)):
    row = data[i]
    if len(row) == 3:
        continue
    else:
        bt = results_bt[i]
        row.append(bt)

with open(os.path.join("data_in",args.task,"train_chat_new.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
