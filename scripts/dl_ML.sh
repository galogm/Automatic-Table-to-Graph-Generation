dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/movielens/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/movielens/raw" && \
[ -f "tmp/ml-latest-small.zip" ] || (mkdir -p tmp && wget --no-check-certificate -O tmp/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  && \
cp tmp/ml-latest-small.zip "$dataset_path/movielens/raw/" && \
unzip -o "$dataset_path/movielens/raw/ml-latest-small.zip" -d "$dataset_path/movielens/raw" && \
mkdir -p "$dataset_path/movielens/old" && \
mkdir -p "$dataset_path/movielens/expert" && \
mkdir -p "$dataset_path/movielens/old/data" && \
mkdir -p "$dataset_path/movielens/expert/data" && \
mkdir -p "$dataset_path/movielens/old/ratings" && \
mkdir -p "$dataset_path/movielens/expert/ratings" && \
python3 -u -m main.preprocessing_dataset mvls $dataset_path && echo DONE " > logs/dl_ML.log  2>&1 & echo $!