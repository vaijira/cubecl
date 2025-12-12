/*https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/08/Kerr_Campbell_Richards_QRD_on_GPUs.pdf?bid=462


Algorithm 2 Computation of W and Y from V and B [5]
1: Y = V (1 : end, 1)
2: W = −B(1) · V (1 : end, 1)
3: for j = 2 to r do
4: v = V (:, j)
5: z = −B(j) · v − B(j) · W Y H v
6: W = [W z]
7: Y = [Y v]
8: end for

Algorithm 3 Block Householder QR
Require: A ∈ C
m×n
, Q
HQ = I
1: Q ← I
2: for k = 1 to n/r do
3: s = (k − 1) · r + 1
4: for j = 1 to r do
5: u = s + j − 1
6: [v, β] = house(A(u : m, u))
7: A(u : m, u : s + r − 1) = A(u : m, u : s + r − 1) −
βvvHA(u : m, u : s + r − 1)
8: V (:, j) = [zeros(j − 1, 1); v]
9: B(j) = β
10: end for
11: Y = V (1 : end, 1)
12: W = −B(1) · V (1 : end, 1)
13: for j = 2 to r do
14: v = V (:, j)
15: z = −B(j) · v − B(j) · W Y Hv
16: W = [W z]
17: Y = [Y v]
18: end for
19: A(s : m, s + r : n) = A(s : m, s + r : n) + Y W HA(s :
m, s + r : n)
20: Q(1 : m, s : m) = Q(1 : m, s : m) + Q(1 : m, s :
m)W Y H
21: end for
*/