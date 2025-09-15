# AutoG & AutoM

# movielens
tsk=ratings
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autom movielens data autog-s type.txt $tsk && python3 -u -m main.rdb MVLS ./data/movielens/autom/$tsk/final ./data/movielens/autom/$tsk/final r2n sage $tsk -c configs/movielens/sage-r2ne.yaml" > logs/movielens-autom-$tsk-final.log &





# AutoG

# avs
tsk=repeater
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog avs data autog-s type.txt $tsk && python3 -u -m main.rdb AVS ./data/avs/autog/$tsk/final ./data/avs/autog/$tsk/final/ r2n sage $tsk -c configs/avs/oracle-$tsk.yaml" > logs/avs-$tsk-final-autog_eval.log  2>&1 &


# retailrocket
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog retailrocket data autog-s type.txt cvr && python3 -u -m main.rdb RR ./data/retailrocket/autog/cvr/final/ ./data/retailrocket/autog/cvr/final/ r2n sage cvr -c configs/rr/oracle-cvr.yaml" > logs/retailrocket-cvr-final-autog_eval.log 2>&1 &


# diginetica
tsk=purchase
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog diginetica data autog-s type.txt $tsk && python3 -u -m main.rdb DIG ./data/diginetica/autog/$tsk/final/ ./data/diginetica/autog/$tsk/final/ r2ne sage $tsk -c multi-table-benchmark/hpo_results/diginetica/$tsk/r2ne-sage.yaml" > logs/diginetica-$tsk-final.log 2>&1 &
tsk=ctr
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog diginetica data autog-s type.txt $tsk && python3 -u -m main.rdb DIG ./data/diginetica/autog/$tsk/final/ ./data/diginetica/autog/$tsk/final/ r2ne sage $tsk -c multi-table-benchmark/hpo_results/diginetica/$tsk/r2ne-sage.yaml" > logs/diginetica-$tsk-final.log 2>&1 &


# ieeecis
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog ieeecis data autog-s type.txt fraud && python3 -u -m main.rdb IEEE ./data/ieeecis/autog/fraud/final/ ./data/ieeecis/autog/fraud/final/ r2n sage fraud -c configs/ieee-cis/sage.yaml" > logs/ieeecis-fraud-final.log  2>&1 &


# mag
tsk=venue
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2ne sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &
tsk=cite
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2ne sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &
tsk=year
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2n sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &



# stackexchange
tsk=churn
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog stackexchange data autog-s type.txt $tsk && python3 -u -m main.rdb STE ./data/stackexchange/autog/$tsk/final ./data/stackexchange/autog/$tsk/final/ r2n sage $tsk -c configs/stackexchange/sage.yaml" > logs/stackexchange-$tsk-final.log 2>&1 &
tsk=upvote
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog stackexchange data autog-s type.txt $tsk && python3 -u -m main.rdb STE ./data/stackexchange/autog/$tsk/final ./data/stackexchange/autog/$tsk/final/ r2n sage $tsk -c configs/stackexchange/sage.yaml" > logs/stackexchange-$tsk-final.log 2>&1 &


# Outbrain
tsk=ctr
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog outbrain data autog-s type.txt $tsk && python3 -u -m main.rdb OBS ./data/outbrain/autog/$tsk/final ./data/outbrain/autog/$tsk/final/ r2n sage $tsk -c configs/outbrain-small/oracle-$tsk.yaml" > logs/outbrain-$tsk-final.log &


# movielens
tsk=ratings
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog movielens data autog-s type.txt $tsk && python3 -u -m main.rdb MVLS ./data/movielens/autog/$tsk/final ./data/movielens/autog/$tsk/final r2n sage $tsk -c configs/movielens/sage-r2ne.yaml" > logs/movielens-$tsk-final.log &

