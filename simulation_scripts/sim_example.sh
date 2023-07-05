#!/bin/bash
alpha=0.6
m0=0.4
m1=$(echo "1-$m0"|bc)
for mind2 in 5 6 7
do
for mind1 in 5.00 5.25 5.50 5.75 6.00 6.25 6.50 6.75 7.00
do
maxd1=$mind1
maxd2=$mind2
    time python sim-multilayer.py -modelname mask -itemname es -n 50000 -e 1000 -cp 100 -alpha $alpha -m $m0 $m1 -T 0.6 0.5 -tm1 0.2 0.5 -tm2 0.3 0.5 -mind $mind1 $mind2 -maxd $maxd1 $maxd2 -ns 1 1 -kmax 20 20 -nc 100 -change 0 -msg exp30_1_${mind1}_$mind2 #-mdl1 9 -mdl2 10 
    aklog
done
done