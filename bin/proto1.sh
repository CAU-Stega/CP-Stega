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
data_path="stego_data/fewshot3500_movie_13way.json"
#"stego_data/fewshot500_movie_13way.json"
#"stego_data/fewshot500_news_13way.json"
#"stego_data/fewshot500_twitter_13way.json"
#stego_data/fewshot500_bpw_10way.json#7,4,4
n_train_class=7
n_val_class=4
n_test_class=4
for shot in 10
do
    if [ "$dataset" = "fewrel" ] && [ "$embedding" = "cnn" ]; then
        python src/main.py \
            --cuda 0 \
            --way 5 \
            --shot 1 \
            --query 25 \
            --mode "train" \
            --embedding $embedding \
            --classifier "proto" \
            --dataset=$dataset \
            --data_path=$data_path \
            --n_train_class=$n_train_class \
            --n_val_class=$n_val_class \
            --n_test_class=$n_test_class \
            --auxiliary pos
    else
        python src/main.py \
            --cuda 1 \
            --way 3 \
            --shot $shot \
            --query 25 \
            --mode "train" \
            --embedding lstmatt \
            --classifier "proto" \
            --dataset=$dataset \
            --data_path=$data_path \
            --n_train_class=$n_train_class \
            --n_val_class=$n_val_class \
            --n_test_class=$n_test_class
    fi
done
