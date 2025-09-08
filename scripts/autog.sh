## movielens
nohup python3 -u -m main.autog movielens data autog-s type.txt ratings > logs/movielens-ratings.log 2>&1 &

nohup python3 -u -m main.autog outbrain data autog-s type.txt ctr > logs/outbrain.log 2>&1 &

nohup python3 -u -m main.autog retailrocket data autog-s type.txt cvr > logs/retailrocket.log 2>&1 &

nohup python3 -u -m main.autog stackexchange data autog-s type.txt upvote > logs/stackexchange.log 2>&1 &


## mag venue
nohup python3 -u -m main.autog mag data autog-s type.txt venue > logs/mag-venue.log 2>&1 &

nohup python3 -u -m main.autog mag data autog-s type.txt cite > logs/mag-cite.log 2>&1 &

nohup python3 -u -m main.autog mag data autog-s type.txt cite > logs/mag-cite.log 2>&1 &

## after run this, you probably need to delete the round 0 directory otherwise there will be issues



## ieee-cis
nohup python3 -u -m main.autog ieeecis data autog-s type.txt fraud > logs/ieee-cis-fraud.log 2>&1 &



## avs
nohup python3 -u -m main.autog avs data autog-s type.txt repeater > logs/avs.log 2>&1 &

## diginetica
nohup python3 -u -m main.autog diginetica data autog-s type.txt purchase > logs/diginetica-purchase.log 2>&1 &

nohup python3 -u -m main.autog diginetica data autog-s type.txt ctr > logs/diginetica-ctr.log 2>&1 &