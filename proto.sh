#dataset=amazon
#data_path="data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path="data/fewrel.json"
#n_train_class=65
#n_val_class=5
#n_test_class=10

#dataset=20newsgroup
#data_path="data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7

#dataset=huffpost
#data_path="data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16

#dataset=rcv1
#data_path="data/rcv1.json"
#n_train_class=37
#n_val_class=10
#n_test_class=24

dataset=stego
data_path="stego_data/fewshot3500_twitter_13way.json"
#"stego_data/fewshot500_movie_13way.json"
#"stego_data/fewshot500_news_13way.json"
#"stego_data/fewshot500_twitter_13way.json"
#stego_data/fewshot500_bpw_10way.json#7,4,4
n_train_class=7
n_val_class=4
n_test_class=4
pretrained_bert='bert-base-uncased'
# For P-MAML, use need to replace pretrained_bert by the output of
# https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
# Otherwise, MAML will be learned from bert_base_uncased
bert_cache_dir='bert-base-uncased/'


for shot in 10 
do
        python src/main.py \
        --cuda 1 \
        --way 3 \
        --shot $shot \
        --query 25 \
        --mode "train" \
        --bert \
        --pretrained_bert $pretrained_bert \
        --bert_cache_dir $bert_cache_dir \
        --embedding lstmatt \
        --classifier "proto" \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class
done
