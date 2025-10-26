# # install python 3.9 use pyenv
# USE_SSH=true curl https://pyenv.run | bash
# sudo apt-get install zlib1g-dev libffi-dev libreadline-dev libssl-dev libsqlite3-dev libncurses5 libncurses5-dev libncursesw5 lzma liblzma-dev libbz2-dev


# pyenv install 3.9
# pyenv local 3.9

# create and activate virtual environment
if [ ! -d '.env' ]; then
    python3 -m venv .env && echo create venv
else
    echo env exists
fi

source .env/bin/activate

# # update pip
python3 -m pip install -U pip

# # torch cuda
python3 -m pip install "torch==1.13.1" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  -c constraints.txt
# python3 -m pip install torch_scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html  --no-deps

# # dgl cuda
python3 -m pip install "dgl==2.1a240205" -f https://data.dgl.ai/wheels-test/cu117/repo.html  -c constraints.txt

# # install requirements
python3 -m pip install -r requirements.txt -c constraints.txt

echo install requirements successfully!
