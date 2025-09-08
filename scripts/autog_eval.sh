# avs
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 -u -m main.autog avs data autog-s type.txt repeater && python3 -u -m main.rdb AVS ./data/avs/autog/final ./data/avs/autog/final/ r2n sage repeater -c configs/avs/oracle-repeater.yaml > logs/avs-repeater-final-autog_eval.log  2>&1 &


# retailrocket
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog retailrocket data autog-s type.txt cvr && python3 -u -m main.rdb RR ./data/retailrocket/autog/cvr/final/ ./data/retailrocket/autog/cvr/final/ r2n sage cvr -c configs/rr/oracle-cvr.yaml" > logs/retailrocket-cvr-final-autog_eval.log 2>&1 &


# diginetica
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog diginetica data autog-s type.txt purchase && python3 -u -m main.rdb DIG ./data/diginetica/autog/purchase/final/ ./data/diginetica/autog/purchase/final/ r2ne sage purchase -c multi-table-benchmark/hpo_results/diginetica/purchase/r2ne-sage.yaml" > logs/diginetica-purchase-final.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog diginetica data autog-s type.txt ctr && python3 -u -m main.rdb DIG ./data/diginetica/autog/ctr/final/ ./data/diginetica/autog/ctr/final/ r2ne sage ctr -c multi-table-benchmark/hpo_results/diginetica/ctr/r2ne-sage.yaml" > logs/diginetica-ctr-final.log 2>&1 &


# ieeecis
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog ieeecis data autog-s type.txt fraud && python3 -u -m main.rdb IEEE ./data/ieeecis/autog/fraud/final/ ./data/ieeecis/autog/fraud/final/ r2n sage fraud -c configs/ieee-cis/sage.yaml" > logs/ieeecis-fraud-final.log  2>&1 &


# mag
tsk=venue
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2ne sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &
tsk=cite
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2ne sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &
tsk=year
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash -c "python3 -u -m main.autog mag data autog-s type.txt $tsk && python3 -u -m main.rdb MAG ./data/mag/autog/$tsk/final ./data/mag/autog/$tsk/final/ r2ne sage $tsk -c configs/mag/oracle-$tsk.yaml" > logs/mag-$tsk-final.log  2>&1 &