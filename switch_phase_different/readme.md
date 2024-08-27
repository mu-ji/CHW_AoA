This folder holds experimental data on the effect of switching on the antenna phase.

-------------------------------------------------------------
The experiemnt data should be named as xxxx_aa_xx_xx_data.npy
xxxx switch pattern id: Different id correspond to different switch patterns
aa : the length of CTE
bb : the distance between tx and rx
cc : the x angle between tx and rx
dd : the y angle between tx and rx
ee : the length of slot

----------------------------------------------------------------
The antenna id:
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15

The antenna rotation:
270 270 270 180
0   270 180 180
0   0   90  180
0   90  90  90
---------------------------------------------------------------
switch pattern id:
0000    :   0 1 2 3 '''''' 13 14 15
0001    :   0 4 8 12 1 5 '''''' 3 7 11 15
0002    :   8 4 9 12 14 13 10 15 7 11 6 3 1 2 5 0
0003    :   0 4 8 12
0004    :   0 1 2 3
0005    :   0 1 0 1 ''''''

-----------------------------------------------------------
remark of each npy:

0000_72_0.5_90_90_1.npy     (100,16,8) 100 times measurement, in each measurement have 16 antenna and 8 samples for each ant
