import os
import openai
import backoff  # for exponential backoff
import openai_manager
import pandas as pd
import numpy as np
import re
import pickle
import argparse
from openai.embeddings_utils import get_embedding, cosine_similarity
#Your API key
openai_manager.append_auth_from_config(config_path='openai_api_config.yml')

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embedding_backoff(*args, **kwargs):
    return get_embedding(*args, **kwargs)
parser = argparse.ArgumentParser()
parser.add_argument("--refinery",action='store_true', default=False)
args = parser.parse_args()
#embedding of self-generated demonstrations
df = pd.read_csv('embedding/embedding_mr_gen.csv')
df['embeddings'] = df.embeddings.apply(eval).apply(np.array)
#embedding of the unlabeled training dataset
df_train = pd.read_csv('embedding/embedding_mr_train.csv')
df_train['embeddings'] = df_train.embeddings.apply(eval).apply(np.array)

if args.refinery:
    print("in refinery annotation")
    with open("../self_training_slm/feedback/right_list_mr.pkl", "rb") as f:
        right_list_all = pickle.load(f) #clean sample idx
        
    with open("../self_training_slm/feedback/demo_index_mr.pkl", "rb") as f:
        top_indices = pickle.load(f) #demonstration retrieval by SLM's embeddings
        
    with open("../self_training_slm/feedback/pred_label_mr.pkl", "rb") as f:
        pred_labels = pickle.load(f) #pseudo-labels by SLM
else:
    print("in initial annotation")
start_idx = 0
test_num = len(df_train)-start_idx
sel_list = range(start_idx,start_idx+test_num)
df_train  = df_train.iloc[sel_list].reset_index(drop=True)
all_text = df_train['text']
all_embeddings = df_train['embeddings']


def askChatGPT(messages):
    MODEL = "gpt-3.5-turbo"
    response =  openai_manager.ChatCompletion.create(
        model=MODEL,
        messages = messages,
        n=1)
    return response

def search_intext(embedding,n=3):
    query_embedding = embedding
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    intext_results = results.head(n)
    return intext_results

all_response = []
all_sentences = []
batch_messages = []
batch_sentences = []
batch_index = []
count = 0
initial_tag = True
for i in sel_list:
    n = 10
    sentence = all_text[i]
    embedding = all_embeddings[i]
    if not args.refinery:
        #initial round: retrieval by bert embeddings
        intext_results = search_intext(embedding,n)
    else:
        #refinery round: retrieval by SLM
        ds_examples = df_train.iloc[top_indices[i]].reset_index(drop=True)
        ds_annos = [pred_labels[idx] for idx in top_indices[i]]
 
    sentence = sentence.replace('"', '\"').replace("\n", "")
    sentence_to_chatgpt = "Given a movie review: '" + sentence + "'"
    
    sentences_example = []
    annos_example = []
    temp_messages=[
            {"role": "system","content": "You are a helpful assistant for the task of text classification on the MR (Movie Review) dataset. You reply with brief, to-the-point answers with no elaboration as truthfully as possible. MR (Movie Review) dataset  is used in sentiment-analysis experiments and this dataset contains movie-review documents labeled with respect to their overall sentiment polarity  (positive or negative). Your task is to a binary classification to classify a movie review as positive or negative according to their overall sentiment polarity. The category is divided into two types: 'positive' and 'negative'."},             
           ]
    for j in range(n):
        if args.refinery:
            temp_sentence = ds_examples.iloc[j]['text'].replace("\n", "").replace('"', '\"')
            temp_anno = "positive" if (ds_annos[j]) else "negative"
        else:
            temp_sentence = intext_results.iloc[j]['text'].replace("\n", "").replace('"', '\"')
            temp_anno = (intext_results.iloc[j]['anno'])
        sentences_example.append(temp_sentence)
        annos_example.append(temp_anno)
        temp_messages.append({"role": "user","content": "Given a movie review: '"+ temp_sentence +"'"})
        temp_messages.append({"role": "user", "content": "How do you feel about the sentiment polarity of the given movie review, is this positive or negative? please answer in a single line with 'positive' or 'negative'. You must answer with 'positive' or 'negative', do not answer anything else."})
        temp_messages.append({"role": "assistant","content": temp_anno})

    temp_messages.append({"role": "user", "content": sentence_to_chatgpt})
    temp_messages.append({"role": "user", "content": "How do you feel about the sentiment polarity of the given movie review, is this positive or negative? please answer in a single line with 'positive' or 'negative'. You must answer with 'positive' or 'negative', do not answer anything else."})
    # print(temp_messages)
    batch_index.append(i)
    batch_messages.append(temp_messages)
    batch_sentences.append(sentence)
    count += 1
    if count == 100: # set a budget to reduce the negative impact of exceptions in annotations
        response = askChatGPT(batch_messages)
        all_response += (response)
        all_sentences += batch_sentences
        for i in range(len(batch_sentences)):
            final_anno = response[i]['choices'][0]['message']['content']
            if final_anno[-1] == '.':
                final_anno = final_anno[:-1]
            if final_anno.lower() == "negative":
                final_label = '0'
            elif final_anno.lower() == "positive":
                final_label = '1'
            else:
                final_label = '-1'
            if args.refinery and batch_index[i] in right_list_all:
                final_label = str(pred_labels[batch_index[i]])
            if initial_tag:
                with open('results/output_mr_train.txt', 'w') as f_out:
                    f_out.write(final_label+'\n')
                    initial_tag = False
            else:
                with open('results/output_mr_train.txt', 'a') as f_out:
                    f_out.write(final_label+'\n')
                
        batch_messages = []
        batch_sentences = []
        batch_index = []
        count = 0
if len(batch_messages):
    response = askChatGPT(batch_messages)
    all_response += response
    all_sentences += batch_sentences
    for i in range(len(batch_sentences)):
        final_anno = response[i]['choices'][0]['message']['content']
        if final_anno[-1] == '.':
            final_anno = final_anno[:-1]
        if final_anno.lower() == "negative":
            final_label = '0'
        elif final_anno.lower() == "positive":
            final_label = '1'
        else:
            final_label = '-1'
        if args.refinery and batch_index[i] in right_list_all:
            #clean samples do not require reannotation
            final_label = str(pred_labels[batch_index[i]])


        if initial_tag:
            with open('results/output_mr_train.txt', 'w') as f_out:
                f_out.write(final_label+'\n')
                initial_tag = False
        else:
            with open('results/output_mr_train.txt', 'a') as f_out:
                f_out.write(final_label+'\n')
            
