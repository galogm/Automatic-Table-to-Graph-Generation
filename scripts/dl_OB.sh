dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/outbrain/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/outbrain/raw" && \
mkdir -p "$dataset_path/outbrain/old" && \
mkdir -p "$dataset_path/outbrain/expert" && \
mkdir -p "$dataset_path/outbrain/old/data" && \
mkdir -p "$dataset_path/outbrain/expert/data" && \
mkdir -p "$dataset_path/outbrain/old/ctr" && \
mkdir -p "$dataset_path/outbrain/expert/ctr" && \
python3 -u -m dbinfer.main download outbrain-small && \
python3 -u -m main.preprocessing_dataset outbrain && echo DONE " > logs/dl_OB.log  2>&1 & echo $!