
## 2 Duality <n>6</n>

### 问题描述

>Please write the dual problem of the **MultiCommodityFlow** problem in *Lec8.pdf*, and give an explanation of the dual variables.
>>Please also construct an instance, and try to solve both primal and dual problem using $GLPK$ or $Gurobi$ or other similar tools.
>
> **MultiCommodityFlow problem**
> **Input:**
> A directed graph $G =< V, E >$. Each edge e has a capacity $C_e$. A total of $k$ commodities, and for commodity $i$, $s_i$, $t_i$, and $d_i$ denote the source, sink, and demand, respectively.
> **Output:**
> A feasible flow for commodity $i$ (denoted as $f_i$) satisfying the $Flow-Conservation$, and capacity constraints, i.e. the aggregate flow on edge $e$ cannot exceed its capacity $C_e$.

### 原始问题

$$
\large 
\begin{array}{rrcll}
max & 0 \\\
s.t. & \sum_{i=0}^k f_i(u,v) &\le &c(u,v) & \text{for each } u,v \in V \\\
& \sum_{u,(u,v)\in E} f_i(u,v) & = &\sum_{w,(v,w)\in E}f_i(v,w) & \text{for each } i,v \in V - \{s_i,t_i\} \\\
& \sum_{v,(s_i,v)\in}f_i(s_i,v) &= &d_i & \text{for each } i \\\
& f_i(u,v) &\ge &0 & \text{for each } i,(u,v) \\\
\end{array}
$$

### 对偶问题

