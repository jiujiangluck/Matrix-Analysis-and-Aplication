
# Preface

写在前面的话

此笔记是基于`某个笔记`增删该而来，如有侵权，请联系我哦! 当然有错误也可以联系我哦！

目前此笔记尚未开源，如有用于商业用途请联系我！

联系方式：[发送邮件](Mailto:1828390107@qq.com)

# Background
##  Vector

- 线性相关
存在一组 $a_1,a_2,...a_k$ 不全为零的数，使得 $a_1 \vec x_1 + a_2 \vec x_2 + ... + a_k \vec x_k = \vec 0$ ，那么可以称这组向量 $\vec x_1,\vec x_2,...,\vec x_k$ 是线性相关的。

- 线性无关
当且仅当 $a_1,a_2,...a_k$ 全都为0时， $a_1 \vec x_1 + a_2 \vec x_2 + ... + a_k \vec x_k = \vec 0$ 才成立，，那么可以称这组向量 $\vec x_1,\vec x_2,...,\vec x_k$ 是线性无关的。

- 极大线性无关组
如果线性无关的 $\vec x_1,\vec x_2,...,\vec x_k$ 是向量组 $\vec x$ 部分组，且 $\vec x$ 中任一向量都可以用 $\vec x_1,\vec x_2,...,\vec x_k$ 表示，那么 $\vec x_1,\vec x_2,...,\vec x_k$ 就是一个极大线性无关组或最大线性无关组。

- 向量运算
内积 $\vec a \cdot \vec b = \sum a_i b_i$
叉积 $\vec a \times \vec b=\left[\begin{matrix}\vec i & \vec j  & \vec k\\ a_x & a_y & a_z\\ b_x & b_y & b_z\end{matrix}\right]$

- 范数

    - $\left\| {x} \right\| _1 = \sum|x_i|$
    - $\left\| {x} \right\| _2 = \sqrt{\sum_{i} x_{i}^2}$
    - $\left\| {x} \right\| _{\infty} = \max\{|x_i|\}$
    - $\left\| {x} \right\| _p = (\sum|x_i|^p)^{1 \over p}$
    - 向量范数的性质：1) 非负性；2) 齐次性；3) 三角不等性

##  Matrix

- 矩阵转置

$$A^T =\left[\begin{matrix} a_{11} & a_{21} & \cdots & a_{n1}\\ a_{12} & a_{22} & \cdots & a_{n2}\\ \vdots & \vdots & \ddots & \vdots\\ a_{1n}  & a_{2n} & \cdots & a_{nn}\end{matrix}\right]$$

$$(AB)^T =B^TA^T$$

$$(A+B)^T=A^T+B^T$$

$$(kA)^T=kA^T$$

- 共轭转置
$$A^H = (\bar A)^T$$
eg. $$ \left(\begin{matrix} 1 & 2+i \\ 1-i & 2 \end{matrix}\right)^H = \left(\begin{matrix} 1 & 1+i \\ 2-i & 2 \end{matrix}\right)$$
酉矩阵：复数域上的正交矩阵$u_i^H u_j=\left\{\begin{matrix}
0, &  i\neq j\\ 
1, & i=j
\end{matrix}\right.$

- 伴随矩阵
$$A^* = \left[\begin{matrix}A_{11} & A_{21} & \cdots & A_{n1}\\ A_{12} & A_{22} & \cdots & A_{n2}\\ \vdots & \vdots & \ddots & \vdots\\ A_{1n}  & A_{2n} & \cdots & A_{nn}\end{matrix}\right]$$
$A_{ij}=(-1)^{i+j}M_{ij}$

- 矩阵的迹

$$tr(A)=\sum_i^na_{ii}=\sum_i^n\lambda_i$$

$$a=tr(a)$$

$$tr(AB)=tr(BA)$$

$$tr(A+B)=tr(A)+tr(B)$$

$$tr(A)=tr(A^T)$$

$$tr(A^TB)=\sum_{i,j}A_{ij}B_{ij}$$

$$tr(A^T(B \odot C))=tr((A \odot B)^TC)$$

- 范数
$$\left\| A \right\|_F = \sqrt{\sum_{i,j} a_{ij}^2}=tr(AA^T)$$

$$\left\| A \right\|_2 = \sqrt{\lambda_{max}(A^TA)}=\delta_{max}(A)$$

$$\left\| A \right\|_1 = \max_{j} \sum _{i=1}^n|a_{ij}|$$

$$\left\| A \right\|_\infty = \max_{i} \sum _{j=1}^n|a_{ij}|$$

$$\left\| A \right\|_* = \sum \delta(A)$$

$$\left\| A \right\|_p = \max_{\left\| x \right\|_p = 1} \left\| Ax \right\|_p$$

- 标准正交矩阵$U = [\alpha_1, \alpha_2, \cdots, \alpha_n ]$
$$\alpha_i^T\alpha_j=\left\{\begin{matrix}
1, &  i\neq j\\ 
0, & i=j
\end{matrix}\right.$$
满足如下性质：
    1) $U^{-1}=U^T$
    2) $rank(U)=n$
    3) $U^TU=UU^T=E$
    4) $\left\| U \cdot A \right\| = \left\| A \right\| $


- 部分列正交矩阵$U=[U_1, U_2]$
$U_1 = [u_1, u_2, \cdots, u_r] \in R^{n \times r}$
$u_i^T u_j=\left\{\begin{matrix}
0, &  i\neq j\\ 
1, & i=j
\end{matrix}\right.$
$U_1^T U_1 = E_{r \times r}$
$UU^T = [U_1, U_2]\left ( \begin{matrix} U_1\\ U_2 \end{matrix}\right )= U_1 U_1^T+ U_2 U_2^T = E $
$U_2$是正交补

- 正交化
有一组向量$a_1, a_2, \cdots, a_n$ 寻找$q_1, q_2, \cdots, q_n$使得
$$span\{a_1, a_2, \cdots, a_n\}=span\{q_1, q_2, \cdots, q_n\}$$
且$Q=[q_1, q_2, \cdots, q_n]$是标准正交矩阵
`Gram-Schmidt`
    1) $span\{q_1\}=span\{a_1\}$， 则$q_1={a_1 \over \left\| a_1 \right \|}$
    2) 假设$span\{a_1, a_2, \cdots, a_k\}=span\{q_1, q_2, \cdots, q_k\}$且满足$q_i \perp q_j, i \neq j, \left\|q_i\right\|=1$
    3) 如何构造$q_{k+1}$使其满足：$\left\{ \begin{aligned} & span\{q_1, q_2, \cdots, q_k\} \oplus span\{q_{k+1}\} = span\{a_1, a_2, \cdots, a_{k+1}\}\\ & q_{k+1} \perp q_i, i = 1,2, \cdots, k\\ & \left\| q_{k+1} \right\| =1 \end{aligned} \right.$
    $a_{k+1}=\sum_{i=1}^{k+1}r_{i,k+1} q_i 
    \Rightarrow q_i^Ta_{k+1}=r_{i,k+1}q_i^Tq_i
    $
    \
    $q_{k+1} = {{a_{k+1} - \sum_{i=1}^{k}r_{i,k+1}q_i} \over {\left\| a_{k+1} - \sum_{i=1}^{k}r_{i,k+1}q_i \right \|}}$, 其中$r_{i,k+1}=q_i^Ta_{k+1}, i=1,2,3,\cdots,k$
从`QR Decomposition`看`Gram-Schmidt`：
$$[a_1, a_2, \cdots, a_n]=[q_1, q_2, \cdots, q_n]\left[\begin{matrix} r_{11} & r_{12} & \cdots & r_{1n}\\ 0 & r_{22} & \cdots & r_{2n}\\ \vdots & \vdots & \ddots & \vdots\\ 0  & 0 & \cdots & r_{nn}\end{matrix}\right]$$
$$\Rightarrow \left\{ \begin{aligned} & a_1 = r_{11}q_1\\ & a_2  = r_{12}q_1 + r_{22}q_2 \\ & \vdots  \end{aligned} \right. \Rightarrow \left\{ \begin{aligned} & q_1 = {a_1 \over \left\| a_1 \right \|}\\ & q_2 = {a_2 - r_{12}q_1 \over \left\| a_2 - r_{12}q_1 \right \|} \\ & \vdots  \end{aligned} \right.$$
`Arnoldi`分解：在$\Kappa_k(A,r_0)=span\{r_0,Ar_0,\cdots,A^{k-1}r_0\}$上运用`Gram-Schmidt`
        1) 令$v_1={r_0 \over \left\| r_0 \right \|}$

        2) 假设已构造$\{v_1, v_2, \cdots, v_k\}, v_k \in \Kappa_k(A,r_0), q_i \perp q_j, i \neq j, \left\|v_i\right\|=1$

        3) 如何构造$v_{k+1}$使其满足：$\left\{ \begin{aligned} & v_{k+1} \in \Kappa_{k+1}(A,r_0) \\ & v_{k+1} \perp v_i, i = 1,2, \cdots, k\\ & \left\| v_{k+1} \right\| =1 \end{aligned} \right.$
        $v_{k+1} \in \Kappa_{k+1}(A,r_0) = \Kappa_{k}(A,r_0) \oplus span\{Av_k\} = span\{v_1, v_2, \cdots, v_{k+1}\}$
        $Av_k = \sum_{i=1}^{k+1}h_{i,k} v_i \Rightarrow v_i^TAv_k=h_{i,k}v_i^Tv_i$
        $v_{k+1} = {{Av_k - \sum_{i=1}^{k}h_{i,k}v_i} \over {\left\|Av_k - \sum_{i=1}^{k}h_{i,k}v_i \right \|}}$, 其中$h_{i,k}=v_i^TAv_k, i=1,2,3,\cdots,k$

- A-正交
$\left\| X \right\|_A= \sqrt{X^TAX}$
有一组向量$a_1, a_2, \cdots, a_n$ 寻找$q_1, q_2, \cdots, q_n$使得
$$span\{a_1, a_2, \cdots, a_n\}=span\{q_1, q_2, \cdots, q_n\}$$
且$q_1, q_2, \cdots, q_n$与A正交，即$q_i^TAq_j=\left\{\begin{matrix}
\neq 0, &  i= j\\ 
0, & i \neq j
\end{matrix}\right.$
做法同上，这里就不赘述。

- 广义逆

- 标准正交投影

# Matrix Decomposition
$A \in C^{n \times n} , B \in C^{m \times n}$ ，U,V为酉阵

## QR Decomposition
$$A = [a_1, a_2, \cdots, a_n] = Q \cdot R =[q_1, q_2, \cdots, q_n]\left[\begin{matrix} r_{11} & r_{12} & \cdots & r_{1n}\\ 0 & r_{22} & \cdots & r_{2n}\\ \vdots & \vdots & \ddots & \vdots\\ 0  & 0 & \cdots & r_{nn}\end{matrix}\right]$$

## LU Decomposition
$$A =  L \cdot U $$
L为上三角矩阵，U为下三角矩阵
$Ax=b \Rightarrow (LU)x = b \Rightarrow \left\{\begin{matrix}& Ly=b\\& Ux=y
\end{matrix}\right.$

## Shur Decomposition
$$A =  U R U^H $$
其中U是正交阵，R是上三角阵。
若A是`Hermitian`矩阵，则R是对角阵，即$A=U \Lambda U^T$
## Single Value Decompostion
$$B = U \Sigma V^H$$
若$rank(B)=r$， 则有
$$A=\left(u_1, u_2,\cdots, u_m \right) \left( \begin{matrix} \sigma_1 & & & & & \\ & \ddots & & & &\\  & & \sigma_r & & &\\  & & & 0 & &\\  & & & &\ddots &\\  & & & & & 0 \end{matrix} \right)_{m \times n}\left(v_1, v_2,\cdots, v_n \right)^H=U_1\Sigma_1V_1^H$$
其中，$U_1 = \left(u_1, u_2,\cdots, u_r \right), V_1 = \left(v_1, v_2,\cdots, v_r \right)$

因此根据$B = U \Sigma V^H$， 可以得到
$$Av_i=U\Sigma\left( \begin{matrix} v_1^H \\ \vdots \\ v_n^H\end{matrix} \right)v_i=\sigma_i u_i$$
$$u_i^HA=u_i^H\left(u_1, u_2,\cdots, u_m \right) \Sigma V^H=\sigma_i v_i^H$$

思考：什么情况下奇异值与特征值相同？
## Householder Transformation


## Gwens Rotation Transformation

# Matrix Differential

## 导数

- 标量对向量的求导
对于标量$f, \vec x_{(n\times 1)}$ 有：
$$\frac{\partial f}{\partial \vec x}=\left[\partial f \over \partial x_{i}\right] $$

- 标量对矩阵的求导
对于标量$f, X_{(m\times n)}$ 有：
$$\frac{\partial f}{\partial X}=\left[\partial f \over \partial x_{ij}\right] $$
我们知道标量对标量的梯度gradient和微分differentiation有这样的关系：
$$df=f'(x)dx$$
$$df=\sum_i{\partial f \over \partial x_i}dx_i={\frac{\partial f}{\partial \vec x}}^T d\vec x$$
那么标量对矩阵也存在：
$$df=\sum_{ij}{\partial f \over \partial x_{ij}}dx_{ij}=tr({\partial f \over \partial X}^TdX)$$
例子1：已知$f=|X|$，求$df$ ?\
我们知道
$$|X|=\sum_ix_{ij}A_{ij}(A_{ij}代数余子式)$$
将上式代入得：
$$\frac{\partial f}{\partial X}=\left[\partial \sum_kx_{kj}A_{kj} \over \partial x_{ij}\right]=\left[A_{ij}\right]={(X^*)}^T$$
因此有
$$df=tr({\partial f \over \partial X}^TdX)=tr(X^*dX)=|X|tr(X^{-1}dX)$$
例子2：求$dX^{-1}$ ?
我们知道
$$X X^{-1}=E$$
对等式两边微分有
$$dXX^{-1}=dE$$
$$XdX^{-1}=-X^{-1}dX$$
因此有
$$dX^{-1}=-X^{-1}dXX^{-1}$$
例子3：$f = {\vec a}^TX \vec b$，求 $\partial f \over \partial X$?
$$df ={\vec a}^T dX \vec b=tr({\vec a}^T dX \vec b)=tr( \vec b {\vec a}^T dX)=tr({\partial f \over \partial X}^TdX)$$
因此 ${\partial f \over \partial X } =\vec a {\vec b}^T$

- 复合法则
已知$f=g(Y)$，且 $Y=h(X)$，怎么求 $\partial f \over \partial X$?(其中g和h都是逐元素的函数)
$$df=tr({\partial f \over \partial Y}^TdY)=tr({\partial f \over \partial Y}^T(h'(X)\odot dX))=tr(({\partial f \over \partial Y}\odot h'(X))^TdX)$$
例子4：$loss =-{\vec y}^T\log\space softmax(W\vec x )$，求 ${\partial \space loss \over \partial W }$。 $\vec y$ 是只有一个元素为1其余元素为0的向量。
$$softmax(\vec x) = {{e^{\vec x}}\over {{\vec 1}^Te^{\vec x}}}$$
$$loss =-{\vec y}^TW\vec x +({\vec y}^T\vec 1) \log({\vec 1}^Te^{W\vec x}) \tag{*}$$
$$d\space loss = -{\vec y}^TdW\vec x+{{{\vec 1}^T(e^{W\vec x }\odot dW\vec x)}\over{{\vec 1}^Te^{W\vec x}}} \tag{**}$$
$$d\space loss = -{\vec y}^TdW\vec x+{{{(e^{W\vec x }})^TdW\vec x}\over{{\vec 1}^Te^{W\vec x}}}$$
$$d\space loss =tr( -{\vec y}^TdW\vec x+{{{(e^{W\vec x }})^TdW\vec x}\over{{\vec 1}^Te^{W\vec x}}})$$
$$d\space loss = tr(\vec x (softmax(W \vec x) - \vec y)^T dW)$$
$${\partial \space loss \over \partial W } = (softmax(W \vec x) - \vec y){\vec x}^T$$
注意:\
(*)式 $\log({\vec b \over c}) =\log \vec b - \vec 1\log c$，且 ${\vec y}^T\vec 1 = 1$\
(**)式 $\log({\vec 1}^Te^{W\vec x})$ 是标量，$e^{W\vec x}$ 是逐元素函数，因此$d\log({\vec 1}^Te^{W\vec x})={1\over{{\vec 1}^Te^{W\vec x}}}\cdot {{\vec 1}^T(e^{W\vec x }\odot dW\vec x)}$

- 向量对向量求导
对于 $\vec f_{(m\times 1)}, \vec x_{(n\times 1)}$ 有：
$$\frac{\partial \vec f}{\partial \vec x}=\left[\partial f_{i} \over \partial x_{j}\right]_{(n \times m)} $$
$$d\vec f=\left[{\frac{\partial f_i}{\partial \vec x}}^T \right]d\vec x ={\frac{\partial \vec f}{\partial \vec x}}^Td\vec x$$

- 矩阵对矩阵求导
对于矩阵$F_{(m\times n)},X_{(p\times q)}$ 有：
$$\frac{\partial F}{\partial X}=\left[\partial F_{ij} \over \partial x_{kl}\right]_{(pq\times mn)}$$
矩阵向量化：
对于矩阵$X_{(p\times q)}$，其矩阵向量化$vec(X)_{(pq\times 1)}=\left[X_1^T, X_2^T,...,X_q^T\right]^T,X_i是X的列向量$。
$$vec(A+B)=vec(A)+vec(B)$$
$$vec(\vec a {\vec b}^T)=\vec b\otimes \vec a$$
$$X=\sum_iX_i{e_i}^T$$
$$vec((AB)\otimes (CD))=vec((A \otimes C)(B \otimes D))$$
$$vec(AXB)=vec(\sum_iAX_i{e_i}^TB)=\sum_ivec((AX_i)(B^Te_i)^T)=\sum_i(B^Te_i)\otimes(AX_i)=(B^T\otimes A)vec(X)$$
因此有：
$$\frac{\partial F}{\partial X}={\frac{\partial vec(F)}{\partial vec(X)}}_{(pq\times mn)}$$
$$vec(dF)={\frac{\partial F}{\partial X}}^Tvec(dX)$$
求导时矩阵被向量化，弊端是这在一定程度破坏了矩阵的结构，会导致结果变得形式复杂；好处是多元微积分中关于`Gradient`、`Hessian`矩阵的结论可以沿用过来，只需将矩阵向量化。
