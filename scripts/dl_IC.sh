dataset_path="data"

nohup bash -c "mkdir -p $dataset_path/ieeecis/raw && \
([ -f $dataset_path/ieeecis/raw/ieee-fraud-detection.zip ] || kaggle competitions download -c ieee-fraud-detection --path $dataset_path/ieeecis/raw || (echo ERROR: set kaggle token first: https://www.kaggle.com/docs/api && exit 1)) && \
unzip -o $dataset_path/ieeecis/raw/ieee-fraud-detection.zip -d $dataset_path/ieeecis/raw && \
mkdir -p $dataset_path/ieeecis/old && \
mkdir -p $dataset_path/ieeecis/expert && \
mkdir -p $dataset_path/ieeecis/old/data && \
mkdir -p $dataset_path/ieeecis/expert/data && \
mkdir -p $dataset_path/ieeecis/old/fraud && \
mkdir -p $dataset_path/ieeecis/expert/fraud && \
python3 -u -m main.preprocessing_dataset IEEE-CIS $dataset_path && echo DONE" > logs/dl_IC.log  2>&1 & echo $!
