# # install python 3.8 use pyenv
# USE_SSH=true curl https://pyenv.run | bash
# sudo apt-get install zlib1g-dev libffi-dev libreadline-dev libssl-dev libsqlite3-dev libncurses5 libncurses5-dev libncursesw5 lzma liblzma-dev libbz2-dev
# pyenv install 3.8
# pyenv local 3.8

# create and activate virtual environment
if [ ! -d '.env' ]; then
    python3 -m venv .env && echo create venv
else
    echo venv exists
fi

source .env/bin/activate

# # update pip
python3 -m pip install -U pip

# # torch cuda
python3 -m pip install "torch==1.13.1" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# python3 -m pip install torch-scatter==2.1.1 torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# # dgl cuda
python3 -m pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu117/repo.html

# # install requirements
python3 -m pip install -r multi-table-benchmark/conda/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo install requirements successfully!
