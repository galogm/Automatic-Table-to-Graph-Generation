dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/retailrocket/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/retailrocket/raw" && \
mkdir -p "$dataset_path/retailrocket/old" && \
mkdir -p "$dataset_path/retailrocket/expert" && \
mkdir -p "$dataset_path/retailrocket/old/data" && \
mkdir -p "$dataset_path/retailrocket/expert/data" && \
mkdir -p "$dataset_path/retailrocket/old/cvr" && \
mkdir -p "$dataset_path/retailrocket/expert/cvr" && \
mkdir -p "$dataset_path/retailrocket/realold/" && \
mkdir -p "$dataset_path/retailrocket/realold/data" && \
mkdir -p "$dataset_path/retailrocket/realold/cvr" && \
python3 -u -m dbinfer.main download retailrocket && \
python3 -u -m main.preprocessing_dataset RR && echo DONE " > logs/dl_RR.log  2>&1 & echo $!