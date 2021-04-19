# Code and Data for Paper "Improving Cross-Modal Alignment in Vision Language Navigation via Syntactic Information" (NAACL 2021)

## Environment Installation
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

Download Room-Across-Room navigation data and save under /tasks:
```
gsutil -m cp -R gs://rxr-data .
```

Download image features for environments:
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

Python requirements
```
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DOSMESA_RENDERING=ON ..
make -j8
```

## Code

### Parsing
```
python r2r_src/parsing.py
python r2r_src/parsing_hite.py
```
parsing.py parses all English instructions in R2R and RxR. parsing_hite.py parses all Hindi and Telugu instructions in RxR.

### Agent
```
bash run/agent_r2r.bash 0
bash run/agent_rxr.bash 0
```
0 is the id of GPU. It will train the agent and save the snapshot under snap/agent/.

agent_r2r.bash runs the agent on R2R dataset, and agent_rxr.bash runs the agent on RxR dataset.

When train and test on RxR dataset, use parameter --language to pick a single language (`en`, `hi`, `te`).

### Baseline
```
bash run/agent_baseline_r2r.bash 0
bash run/agent_baseline_rxr.bash 0
```
Run this code to replicate the baseline.

Similarly, when train and test on RxR dataset, choose language by setting the parameter --language as `en`, `hi`, `te`.
