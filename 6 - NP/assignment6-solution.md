# 算法设计与分析第六次作业 - NP
201628013229058 洪鑫

## 1. Integer Programming <n>1</n>

> Given an integer $m\times n$ matrix A and an integer $m$-vector $b$, the Integer programming problem asks whether there is an integer $n$-vector $x$ such that $Ax \geq b$. Prove that Integer-programming is in NP-complete.
> 

### 证明思路

可满足问题（SAT）可以在多项式时间内规约为整数规划问题。

### 详细证明

首先，整数规划问题是NP问题。因为对于任意一个实例的解，只需要计算$Ax$的值并与$b$进行比较即可验证解得正确性，计算与比较可以再多项式时间内完成。

然后证明对于任意的NP问题，都可以在多项式时间内规约为一个整数规划问题。因为可满足问题是NP-C类问题，这里我们只需要证明，可满足问题（SAT）可以在多项式时间内规约为整数规划问题。

可满足问题规约到整数规划问题的过程：

1. 将“真”与“1”相对应，“假”与“0”相对应，所有变量满足为[0,1]的整数；
2. 将“或”运算用“加法”代替;
3. 将$\bar x$用$1-x$代替；
4. 整个合区范式的每个子句都为真转换为：所有子句对应的和大于等于1;
5. 整理不等式得到$A$和$b$。

比如，对于可满足问题：
$$(x_1 \lor \lnot x_2)\land(¬x_1 \lor \lnot x_3)\land(x_2 \lor \lnot x_3)$$
可以转换为整数规划问题：
$$
\begin{cases}
&x_1 + 1 - x_2 &\geq 1 \\\
&1 - x_1 + 1 - x_3 &\geq 1 \\\
&x_2 + 1 - x_3 &\geq 1 \\\
&x_1,x_2,x_3 &\geq 0 \\\
&x_1,x_2,x_3 &\leq 1
\end{cases}
$$
调整下形式：
$$
\begin{cases}
\begin{array}{rrrrlr}
&x_1 &- x_2 &&\geq &0 \\\
&- x_1 &&- x_3 &\geq &-1 \\\
&&x_2 &- x_3 &\geq &0 \\\
&x_1 &&&\geq &0\\\
&&x_2 &&\geq &0\\\
&&&x_3 &\geq &0 \\\
&-x_1 &&&\geq &-1\\\
&&-x_2 &&\geq &-1\\\
&&&-x_3 &\geq &-1
\end{array}
\end{cases}
$$
可以写成 $Ax \geq b$的形式，其中：
$$
A=
\begin{pmatrix}
\begin{array}{rrr}
1 & -1 & 0 & \\\
-1 & 0 & -1 & \\\
0 & 1 & -1 & \\\
1 & 0 & 0 & \\\
0 & 1 & 0 & \\\
0 & 0 & 1 & \\\
-1 & 0 & 0 & \\\
0 & -1 & 0 & \\\
0 & 0 & -1 &
\end{array}
\end{pmatrix}
\qquad x=
\begin{pmatrix}
x_1 \\ x_2 \\ x_3
\end{pmatrix}
\qquad b=
\begin{pmatrix}
0 \\ -1 \\ 0 \\ 0 \\ 0 \\ 0 \\ -1 \\ -1 \\ -1
\end{pmatrix}
$$

下面说明，可满足问题存在一个解，当且仅当整数规划问题存在一个解。

如果一个可满足问题存在解，那么必然有一个真假赋值，使得所有的子式都为真，那么对应的整数规划问题也就存在相应的0，1赋值使得所有的不等式成立。<br />
同样道理，相应的整数规划问题存在一个解，那么必然首先满足0，1约束，然后对应子式的和至少为1，相应的0，1赋值转换为真假赋值后，使得可满足问题中所有的子式为真。

综上所述，整数规划问题是NP问题，可满足问题（SAT）可以在多项式时间内转换为整数规划问题，因为已知可满足问题是NP-完全的，所以整数规划也同样是NP-完全的。

## 2. Half-3SAT <n>3</n>

>In the Half-3SAT problem, we are given a 3SAT formula $\phi$ with $n$ variables and $m$ clauses, where $m$ is even. We wish to determine whether there exists an assignment to the variables of $\phi$ such that exactly half the clauses evaluate to false and exactly half the clauses evaluate to true. Prove that Half-3SAT problem is in NP-complete.

### 证明思路

3SAT问题可以在多项式时间内规约为一个Half-3SAT问题。

### 详细证明

显然，半-3SAT问题是NP的。因为通过带入解得赋值后统计子句真假的个数，即可判断解是否满足条件，而这一过程是能在多项式时间内完成的。

3SAT问题规约到半-3SAT的过程：

（对于任意一个3SAT问题，假设其表达式为$\phi$，子句的个数为$n$，我们用$\phi '$表示用于求解半-3SAT问题的表达式。）

1. 将$\phi$中所有的子句都添加到$\phi '$中；
2. $\phi '$中添加$n$个形如$x \land \lnot x \land y$的子句，这$n$个子句完全一致；
3. $\phi '$中添加$2n$个形如$p \land q \land r$的子句，这$2n$个子句完全一致，且$p,q,r$和$\phi$中的变量均不相同。

下面说明3SAT问题有解当且仅当半-3SAT问题有解：

_首先，$\phi '$中含有$4n$个子句。我们注意到，第2步添加的$n$个子句总为真，第3步添加的$2n$个子句要么全为真，要么全为假。_<br />
如果3SAT问题有解，那么根据第1步和第2步，显然$\phi '$中至少有$2n$个子句为真。此时，给$p,q,r$都赋值为假，则剩下的$2n$个子句都为假。显然，对于$\phi '$求解半-SAT问题有解。<br />
如果对应的半-3SAT问题有解，那么$\phi '$有$2n$个子句为真，因为第2步已经有$n$个子句为真，所以第3步添加的$2n$子句只能为假，第1步添加的$n$个子句全为真。所以对$\phi$求解3-SAT问题有解。

综上所述，半-3SAT问题是NP问题，3SAT问题可以在多项式时间内规约为半-3SAT问题，因为已知3SAT问题是NP-完全的，所以半-3SAT也同样是NP-完全的。