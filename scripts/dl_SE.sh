dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/stackexchange/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/stackexchange/raw" && \
mkdir -p "$dataset_path/stackexchange/old"
mkdir -p "$dataset_path/stackexchange/expert"
mkdir -p "$dataset_path/stackexchange/old/data"
mkdir -p "$dataset_path/stackexchange/expert/data"
mkdir -p "$dataset_path/stackexchange/old/churn"
mkdir -p "$dataset_path/stackexchange/expert/churn"
mkdir -p "$dataset_path/stackexchange/old/upvote"
mkdir -p "$dataset_path/stackexchange/expert/upvote"
python3 -u -m dbinfer.main download stackexchange && 
python3 -u -m main.preprocessing_dataset stackexchange && echo DONE " > logs/dl_SE.log  2>&1 & echo $!





