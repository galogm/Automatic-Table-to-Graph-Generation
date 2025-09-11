dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/avs/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/avs/raw" && \
mkdir -p "$dataset_path/avs/old" && \
mkdir -p "$dataset_path/avs/expert" && \
mkdir -p "$dataset_path/avs/old/data" && \
mkdir -p "$dataset_path/avs/expert/data" && \
mkdir -p "$dataset_path/avs/old/repeater" && \
mkdir -p "$dataset_path/avs/expert/repeater" && \
python3 -u -m dbinfer.main download avs && \
python3 -u -m main.preprocessing_dataset avs && echo DONE" > logs/dl_AVS.log  2>&1 & echo $!