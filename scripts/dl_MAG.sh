dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/mag/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/mag/raw" && \
mkdir -p "$dataset_path/mag/old" && \
mkdir -p "$dataset_path/mag/expert" && \
mkdir -p "$dataset_path/mag/old/data" && \
mkdir -p "$dataset_path/mag/expert/data" && \
mkdir -p "$dataset_path/mag/old/venue" && \
mkdir -p "$dataset_path/mag/expert/venue" && \
mkdir -p "$dataset_path/mag/old/cite" && \
mkdir -p "$dataset_path/mag/expert/cite" && \
mkdir -p "$dataset_path/mag/old/year" && \
mkdir -p "$dataset_path/mag/expert/year" && \
python3 -u -m dbinfer.main download mag && \
python3 -u -m main.preprocessing_dataset MAG $dataset_path && echo DONE " > logs/dl_MAG.log  2>&1 & echo $!
