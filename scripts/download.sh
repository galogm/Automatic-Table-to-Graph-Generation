bash scripts/dl_AVS.sh
bash scripts/dl_DN.sh
bash scripts/dl_IC.sh
bash scripts/dl_MAG.sh
bash scripts/dl_ML.sh
bash scripts/dl_OB.sh
bash scripts/dl_RR.sh
bash scripts/dl_SE.sh

[ -d "deepjoin" ] ||  sudo apt-get install -y git-lfs && git lfs install && git clone https://github.com/mutong184/deepjoin.git

sudo apt-get install -y graphviz
