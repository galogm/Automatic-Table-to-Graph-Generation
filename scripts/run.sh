# retailrocket

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb RR ./data/retailrocket/autog/final ./data/retailrocket/autog/final/ r2n sage cvr -c configs/rr/oracle-cvr.yaml > logs/retailrocket-cvr-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb RR ./data/retailrocket/expert ./data/retailrocket/expert/ r2n sage cvr -c configs/rr/oracle-cvr.yaml > logs/retailrocket-cvr-expert.log &




## run results for movielens
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MVLS ./data/movielens/autog/final ./data/movielens/autog/final r2n sage ratings -c configs/movielens/sage-r2ne.yaml > logs/movielens-ratings-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MVLS ./data/movielens/expert ./data/movielens/expert r2n sage ratings -c configs/movielens/sage-r2ne.yaml > logs/movielens-ratings-expert.log &




## outbrain OBS

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb OBS ./data/outbrain/autog/final ./data/outbrain/autog/final/ r2n sage ctr -c configs/outbrain-small/oracle-ctr.yaml > logs/outbrain-ctr-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb OBS ./data/outbrain/expert ./data/outbrain/expert/ r2n sage ctr -c configs/outbrain-small/oracle-ctr.yaml > logs/outbrain-ctr-expert.log &





## mag venue

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/autog/final ./data/mag/autog/final/ r2ne sage venue -c configs/mag/oracle-venue.yaml > logs/mag-venue-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/expert ./data/mag/expert/ r2ne sage venue -c configs/mag/oracle-venue.yaml > logs/mag-venue-expert.log &

## mag cite (the performance may be lower, since it's a quick test, to get better performance please switch to better decoder and more early-stop epochs)

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/autog/final ./data/mag/autog/final/ r2ne sage cite -c configs/mag/oracle-cite.yaml > logs/mag-cite-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/expert ./data/mag/expert/ r2ne sage cite -c configs/mag/oracle-cite.yaml > logs/mag-expert.log &

## mag year

## autog-s result, around ~50
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/autog/final ./data/mag/autog/final/ r2n sage year -c configs/mag/oracle-year.yaml > logs/mag-year-final.log &

## autog-a result 
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/autog/round_0 ./data/mag/autog/round_0/ r2n sage year -c configs/mag/oracle-year.yaml > logs/mag-year-round_0.log &

## expert result
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb MAG ./data/mag/expert ./data/mag/expert/ r2n sage year -c configs/mag/oracle-year.yaml > logs/mag-year-expert.log &

# ## ieee-cis needs autog-a and pattern search to get not that bad results


CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb IEEE ./data/ieeecis/autog/expert ./data/ieeecis/expert/ r2n sage fraud -c configs/ieee-cis/sage.yaml > logs/ieeecis-fraud-expert.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u -m main.rdb IEEE ./data/ieeecis/autog/round_0 ./data/ieeecis/autog/round_0 r2n sage fraud -c configs/ieee-cis/oracle-fraud.yaml > logs/ieeecis-fraud-round_0.log &


## avs dataset


## avs autog
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb AVS ./data/avs/autog/final ./data/avs/autog/final/ r2n sage repeater -c configs/avs/oracle-repeater.yaml > logs/avs-repeater-final.log &

## expert

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb AVS ./data/avs/expert ./data/avs/expert/ r2n sage repeater -c configs/avs/oracle-repeater.yaml > logs/avs-repeater-expert.log &


## stackexchange

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb STE ./data/stackexchange/expert ./data/stackexchange/expert/ r2n sage churn -c configs/stackexchange/sage.yaml > logs/stackexchange-churn-expert.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb STE ./data/stackexchange/expert ./data/stackexchange/expert/ r2n sage upvote -c configs/stackexchange/sage.yaml > logs/stackexchange-upvote-expert.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb STE ./data/stackexchange/autog/final ./data/stackexchange/autog/final/ r2n sage churn -c configs/stackexchange/sage.yaml > logs/stackexchange-churn-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb STE ./data/stackexchange/autog/final ./data/stackexchange/autog/final/ r2n sage upvote -c configs/stackexchange/sage.yaml > logs/stackexchange-upvote-final.log &

## diginetica

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb DIG ./data/diginetica/autog/final/ ./data/diginetica/autog/final/ r2ne sage purchase -c multi-table-benchmark/hpo_results/diginetica/purchase/r2ne-sage.yaml > logs/diginetica-purchase-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb DIG ./data/diginetica/expert/ ./data/diginetica/expert/ r2ne sage purchase -c multi-table-benchmark/hpo_results/diginetica/purchase/r2ne-sage.yaml > logs/diginetica-purchase-expert.log &


CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb DIG ./data/diginetica/autog/final/ ./data/diginetica/autog/final/ r2ne sage ctr -c multi-table-benchmark/hpo_results/diginetica/ctr/r2ne-sage.yaml > logs/diginetica-ctr-final.log &

CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.rdb DIG ./data/diginetica/expert/ ./data/diginetica/expert/ r2ne sage ctr -c multi-table-benchmark/hpo_results/diginetica/ctr/r2ne-sage.yaml > logs/diginetica-ctr-expert.log &










