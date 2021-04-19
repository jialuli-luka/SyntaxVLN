name=agent
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --lstm_layer 2
      --tree dep
      --dataset R2R
      --language en
      --batchSize 64
      --subout max --dropout 0.5 --optim rms --lr 1e-4
      --iters 800000
      --maxAction 35
      --maxInput 80"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
