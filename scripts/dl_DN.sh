dataset_path="data"
export DBB_DATASET_HOME="$dataset_path/diginetica/raw"
export DBB_PROJECT_HOME=$dataset_path
nohup bash -c "mkdir -p "$dataset_path/diginetica/raw" && \
mkdir -p "$dataset_path/diginetica/old" && \
mkdir -p "$dataset_path/diginetica/expert" && \
mkdir -p "$dataset_path/diginetica/old/data" && \
mkdir -p "$dataset_path/diginetica/expert/data" && \
mkdir -p "$dataset_path/diginetica/old/ctr" && \
mkdir -p "$dataset_path/diginetica/expert/ctr" && \
mkdir -p "$dataset_path/diginetica/old/purchase" && \
mkdir -p "$dataset_path/diginetica/expert/purchase" && \
python3 -u -m dbinfer.main download diginetica  && 
python3 -u -m main.preprocessing_dataset diginetica"  && echo DONE > logs/dl_DN.log  2>&1 & echo $!