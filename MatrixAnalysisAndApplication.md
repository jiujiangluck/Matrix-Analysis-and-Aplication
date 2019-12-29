
# Preface

写在前面的话

此笔记是基于`某个笔记`增删该而来，如有侵权，请联系我哦! 当然有错误也可以联系我哦！

联系方式：[发送邮件](Mailto:1828390107@qq.com)

最新下载地址:[下载地址](https://github.com/jiujiangluck/Matrix-Analysis-and-Aplication)

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
    - 向量范数的性质：
        1) $\left\| x \right\| \ge 0$
        2) $ \left\| kx \right\|=k\left\| x \right\|$
        3) $ \left\| x + y \right\| \le \left\| x \right\| + \left\| y \right\|$

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
`Hermitian`矩阵：$A^H=A$，例如$U_1U_1^H=(U_1U_1^H)^H$

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
$$\left\| A \right\|_F = \sqrt{\sum_{i,j} a_{ij}^2}=\sqrt{tr(AA^T)}$$
$$\left\| A \right\|_2 = \sqrt{\lambda_{\max}(A^TA)}=\delta_{\max}(A)$$
$$\left\| A \right\|_1 = \max_{j} \sum _{i=1}^n|a_{ij}|$$
$$\left\| A \right\|_\infty = \max_{i} \sum _{j=1}^n|a_{ij}|$$
$$\left\| A \right\|_* = \sum \delta(A)$$
$$\left\| A \right\|_p = \max_{\left\| x \right\|_p = 1} \left\| Ax \right\|_p$$
矩阵范数满足：
    1) $\left\| A \right\| \ge 0$
    2) $ \left\| kA \right\|=k\left\| A \right\|$
    3) $ \left\| A + B \right\| \le \left\| A \right\| + \left\| B \right\|$
    4) $ \left\| AB \right\| \le \left\| A \right\| \cdot \left\| B \right\|$
- 谱半径
$\rho (A)=\max|\lambda(A)|$
    1) 谱半径不是范数
    2) 若A是`Hermitian`矩阵,则$\rho (A)=\left\| A \right\|_2$
    3) $\rho (A)=\inf_{\left\| \cdot \right\|}\left\| A \right\|$
    4) $\sum_kA^k$收敛$\Rightarrow A^k \to 0,\rho (A) \lt 1$
    5) $\left\| A^k \right\|^{1 \over k} \to \rho (A)$

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
对$A=U_1\Sigma_1V_1^H \in C^{m \times n}$, 记$A^+=V_1\Sigma_1^{-1}U_1^H$,满足：
    1) $A^+AA^+=A^+$
    2) $AA^+A=A$
    3) $AA^+,A^+A$都是`Hermitian`

- 正交投影
假设$p$是子空间$W$：$span\{U_1\}$的投影，$p=U_1U_1^H$，$U_1$是酉阵，则有如下性质：
    1) $p^H=p$
    2) $p^2=p$
    3) $\forall x , px \in W$且$(px)^H(x-px)=0$
![](./reflection.png)

- 子空间
    - 子空间距离
        - 点到子空间距离和span{x}到子空间的距离
        具体参考正交投影
        - 两个平面以及同维子空间距离
        $Z,Y$分别是其标准正交基
        $dist(\mathfrak{X},\mathfrak{Y})=\sqrt{1-\sigma_{\min}^2(Z^HY)}$

    - Krylov子空间
    $\Kappa_k(A,r_0)=span\{r_0,Ar_0,\cdots,A^{k-1}r_0\}$
    

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
## Singular Value Decompostion
$$B = U \Sigma V^H$$
若$rank(B)=r$， 则有
$$A=\left(u_1, u_2,\cdots, u_m \right) \left( \begin{matrix} \sigma_1 & & & & & \\ & \ddots & & & &\\  & & \sigma_r & & &\\  & & & 0 & &\\  & & & &\ddots &\\  & & & & & 0 \end{matrix} \right)_{m \times n}\left(v_1, v_2,\cdots, v_n \right)^H=U_1\Sigma_1V_1^H$$
其中，$U_1 = \left(u_1, u_2,\cdots, u_r \right), V_1 = \left(v_1, v_2,\cdots, v_r \right)$

因此根据$B = U \Sigma V^H$， 可以得到
$$Av_i=U\Sigma\left( \begin{matrix} v_1^H \\ \vdots \\ v_n^H\end{matrix} \right)v_i=\sigma_i u_i$$
$$u_i^HA=u_i^H\left(u_1, u_2,\cdots, u_m \right) \Sigma V^H=\sigma_i v_i^H$$

**思考：什么情况下奇异值与特征值相同？**

$A^HA=V \Sigma^H U^HU\Sigma V^H = V\Sigma^H\Sigma V^H$
$AA^H=U\Sigma V^HV \Sigma^H U^H = U\Sigma^H\Sigma U^H$
因此我们可以根据以上两个矩阵的特征值已经特征向量得到矩阵A的奇异值和奇异向量。
例题：$W = \left[ \begin{matrix} 1 & 1 \\ 0 & 1\\ 1 & 0\end{matrix} \right]$
$W^HW=\left[ \begin{matrix} 2 & 1 \\ 1 & 2\end{matrix} \right]$
特征值：3,1
特征向量$v_1=\left[ \begin{matrix} 1\over \sqrt{2}  \\ 1\over \sqrt{2} \end{matrix} \right], v_2=\left[ \begin{matrix} 1\over \sqrt{2}  \\ -{1\over \sqrt{2}} \end{matrix} \right]$
$WW^H=\left[ \begin{matrix} 2 & 1 & 1 \\ 1 & 1 & 0 \\ 1 & 0 & 1\end{matrix} \right]$
特征值：3,1,0
特征向量$u_1=\left[ \begin{matrix} 2\over \sqrt{6}  \\ 1\over \sqrt{6} \\ 1\over \sqrt{6} \end{matrix} \right], u_2=\left[ \begin{matrix} 0 \\ -{1\over \sqrt{2}} \\ 1\over \sqrt{2}   \end{matrix} \right], u_3=\left[ \begin{matrix} -{1\over \sqrt{3}}  \\ 1\over \sqrt{3} \\ 1\over \sqrt{3} \end{matrix} \right]$
因此$W=(u_1,u_2,u_3)\left[ \begin{matrix} 1\over \sqrt{3} & 0  \\ 0 & 1 \\ 0 & 0 \end{matrix} \right] (v_1,v_2)$

## Householder Transformation
要求实现$Hx=y, \left\| x \right\|=\left\| y \right\|$, 要求：$ 1) H^H=H, 2) H是酉阵$
![householder](./householder.png)
$y = x - 2 {{x -y} \over {2}}$
由于$\left\| x \right\|=\left\| y \right\|$， 因此$x,y,x-y$构成等腰三角形
因此根据正交投影可知${{x -y} \over {2}} = px， p=uu^H$是$span\{x-y\}$的正交投影， $u={{x-y}\over{\left\| x -y\right\|}}$
$\therefore H=I-2uu^H$，显然满足上述两点。
利用`Householder Transformation`可以将矩阵稀疏化。
$\left[ \begin{matrix} * & * & * & *  \\ * &  * & * & * \\ 0 & * & * & * \\ 0 & 0 & * & * \end{matrix} \right] \xRightarrow{H} \left[ \begin{matrix} * & * & * & *  \\ 0 &  * & * & * \\ 0 & 0 & * & * \\ 0 & 0 & 0 & * \end{matrix} \right] $
==注意可以跨行和列去选择变换==
$\left[ \begin{matrix} * & 0 & 0  \\ * &  * & 0  \\ * & * & * \\ * & * & * \\ * & * & * \\ * & * & * \end{matrix} \right] \xRightarrow{H} \left[ \begin{matrix} * & 0 & 0  \\ * &  * & 0  \\ * & * & * \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix} \right] $
同样也能用`Householder Transformation`求QR分解

## Givens Rotation Transformation
$\left[ \begin{matrix} C & S \\ -S & C  \end{matrix}\right]\left[ \begin{matrix} a \\ b  \end{matrix}\right] = \left[ \begin{matrix} \sqrt{a^2+b^2} \\ 0  \end{matrix}\right]$，其中$C={{a}\over {\sqrt{a^2+b^2}}}, S={{b}\over {\sqrt{a^2+b^2}}}$

同样也能用`Givens Rotation Transformation`求QR分解
## Application

- 标准正交基
    - $R(A)=\{Ax|x \in R^n\}$
    $\because Ax=U_1 \Sigma_1 V_1^H x=U_1Z$
    $\therefore R(A)=span\{U_1\},R(A^H)=span\{V_1\}$
    - $N(A)=\{x|Ax=0\}$
    $\because Ax=U_1 \Sigma_1 V_1^H x=0 \Rightarrow V_1^H x=0$
    $\therefore N(A)=span\{V_2\},N(N^H)=span\{U_2\}$

- 低秩逼近
设$rank(A)=r, d < r$, 求$\min_{rank(x)=d}=\left\| A - x \right\|_2$。


- 最小二乘法
$\min_x\left\|Ax-b\right\|_2 \Rightarrow \min_{y \in R(A)}\left\|b-y\right\|_2$
$Ax=AA^+b$的通解为齐次解加上特解，其特解为$A^+b$,齐次通解为$\sigma = (I - A^+A)z\in N(A)$
因此$x = A^+b + (I - A^+A)z$


# Matrix Differential

## 导数与微分

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

## 应用
1) 紧紧抓住两个转换公式：$df=tr({\partial f \over \partial X}^TdX), d(trace(f(X)))=trace(df(X))$以及定义，那么几乎所有的导数我们都能求。

2) 泰勒公式 

3) 求最优化问题

    - 最小二乘法：
        - $\min_x \left\| Ax-b \right\|_2^2$

        - $\min_x \left\| Ax-b \right\|_2^2+ \lambda \left\| x\right\|_2^2$

    - 有约束问题： 
        - $\min_x \left\| Ax \right\|_2^2, s.t.\medspace e^Tx=1$或者$s.t.\medspace \left\| x \right\|_2=1$
        - $\min_Utr(U^TAU),s.t.U^HU=I,A$半正定
        - $\min_X[tr(X^TX)-2tr(X)],s.t.XA=\bold{0}$
    - `Locally Linear Embedding`
    给定一组数据$x_i \in R^n$及其邻域数据$x_{i1},x_{i2},\cdots,x_{ij}$, 要求将$x_i$降维为$y_i \in R^d$。
    LLE 认为数据局部是线性的$x_i=\sum_{i=1}^jw_{ij}x_{ij}$，且在降维过程中线性不变,且组合系数不变。
    优化目标：$\arg \min_Y \sum_{i=1}^N\left\| y_i-\sum_{j=1}^kw_{ij}y_{ij}\right\|_2^2,s.t. \medspace YY^T=NI$

4) 总体最小二乘
$$(A+\Delta A)x=(b+\Delta b) \Rightarrow ([A,b]+[\Delta A + \Delta b])\left[ \begin{matrix} {x} \\ -1  \end{matrix}\right]=0$$
令 $B=[A,b], D = [\Delta A + \Delta b], Z=\left[ \begin{matrix} {x} \\ -1  \end{matrix}\right]$
因此要求$x$的近似解即求：$\min_{D,x} \left\| D \right\|_F^2, s.t. \medspace (B+D)Z=0 $
    - 若$B$不是列满秩，则存在$Z\neq 0, BZ=0$，要使$\min_{D,x} \left\| D \right\|_F^2$最小，只需$D=0$
    - 若$B$是列满秩，要使$(B+D)Z=0 $有解，则$r(B+D) \le n$
        - 若B的奇异值满足$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n \ge \sigma_{n+1}$
        要使要使$\min_{D,x} \left\| D \right\|_F^2$最小，只需$D=-\sigma_{n+1}u_{n+1}v_{n+1}^T$
        故$N(B+D)=span\{v_{n+1}\}$
        记$v_{n+1}=(v_{n+1,1},v_{n+1,2},\cdots,v_{n+1,n+1})^T$
        因此$\left( \begin{matrix} {x} \\ -1  \end{matrix}\right) = \left( \begin{matrix} -\frac{v_{n+1,1}}{v_{n+1,n+1}}  \\ -\frac{v_{n+1,2}}{v_{n+1,n+1}} \\ \vdots \\ -1  \end{matrix}\right)$
        - 若B的奇异值满足$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_p \ge \sigma_{p+1} = \cdots = \sigma_{n+1}$
        $N(B+D)=span\{v_{p+1},\cdots,v_{n+1}\}$
        记$V_1=(v_{p+1},\cdots,v_{n+1})^T$
        运用`Householder变换`：
        $$HV_1= \left( \begin{matrix} \hat{v}_{n+1}  \\ 0 \\ \vdots \\ 0  \end{matrix}\right)$$

# Matrix Equation
## 分裂迭代
方程$Ax=b$
令$A=M-N \Rightarrow  Mx=Nx+b$
迭代形式$Mx_{k}=Nx_{k-1}+b$
故$x_k=M^{-1}(Nx_{k-1}+b)$
收敛性要求：$\rho (M^{-1}N) \lt 1$
若A正定，当且仅当$M+N^H$时，$\rho (M^{-1}N) \lt 1$
- Jacobi迭代
$A = D- L -U$
$M = D, N = L + U = D -A$
故$x_k=D^{-1}((L+U)x_{k-1}+b)=(I-D^{-1}A)x_{k-1}+D^{-1}b$
收敛性要求：$\rho (I-D^{-1}A) \lt 1$
适用：A对角占优

- Gauss-Seideld迭代
$A = D- L -U$
$M = D - L , N = U$
故$x_k=(D-L)^{-1}(Ux_{k-1}+b)$
收敛性要求：$\rho ((D-L)^{-1}U) \lt 1$
适用：元素集中在下三角处

- SOR
$M = {1 \over \omega}D - L , N =(({1 \over \omega}-1)D+U)$
故$(D-\omega L)x_k=((1 - \omega)D + \omega U)x_{k-1}+\omega b$
收敛性要求：$\rho ((D-\omega L)^{-1}((1 - \omega)D + \omega U)) \lt 1, 0 \lt \omega \lt 2$
适用：下三角占优

- SSOR
交替迭代
    1) $M = {1 \over \omega}D - L , N =(({1 \over \omega}-1)D+U)$
    2) $M = {1 \over \omega}D - U , N =(({1 \over \omega}-1)D+L)$
    
## 最速下降
$$Ax=b \iff \min_x \psi(x)={1\over 2}(x,x)_A-(b,x)$$
A对称正定
迭代式：$x_{k+1}=x_k + \alpha_k r_k, r_k=b-Ax_k=-\nabla \psi(x_k), \alpha_k=\arg \min_\alpha \left\| x_k + \alpha r_k -x^{*} \right\|$
求$\alpha $只需满足下式：
$$\frac{d\psi (x_k + \alpha r_k)}{d \alpha}=0$$
$$\psi (x_k + \alpha r_k)=\psi(x_k)+\alpha(-r_k,r_k)+\frac{\alpha^2}{2}(Ar_k,r_k)$$
故$(-r_k,r_k)+\alpha(Ar_k,r_k)=0 \Rightarrow \alpha = \frac{\left\|r_k\right\|_2^2}{\left\|r_k\right\|_A^2}$
收敛性分析：${\left\|x_{k+1}-x^{*}\right\|_A}=\min_\alpha \left\|x_k+\alpha r_k-x^{*}\right\|_A=\min_\alpha \left\|x_k+\alpha A(x^{*}-x_k)-x^{*}\right\|_A$
$=\min_\alpha \left\|(I-\alpha A)(x_k-x^{*})\right\|_A $
$\le \min_\alpha \rho (I-\alpha A)\left\|x_k-x^{*}\right\|_A $
$\le \frac{\lambda_n-\lambda_1}{\lambda_n+\lambda_1}\left\|x_k-x^{*}\right\|_A $
$\le (\frac{\lambda_n-\lambda_1}{\lambda_n+\lambda_1})^k\left\|x_0-x^{*}\right\|_A$

## 子空间迭代
$A \in C^{n \times n}$
$Ax=b \iff x=A^{-1}b  \iff A(x^{*}-x_0)=b-Ax_0=r_0 \iff x^{*}=x_0+A^{-1}r_0$
又$A^{-1}=\sum_{i=0}^mC_iA^ir_0$
故$x^{*}=x_0+\Kappa_{m+1}(A,r_0)$
## 共轭CG
$x^{*}=x_0+\Kappa_{m+1}(A,r_0)$
设$g_0,g_1,\cdots,g_{m}$是$\Kappa_{m+1}(A,r_0)$一组标准A正交基，则
$$x^{*}=x_0+\sum_{j=0}^m\alpha_jg_j$$
设$x_{k}=x_0+\sum_{j=0}^{k-1}\alpha_jg_j$，则$x_{k+1}=x_k+\alpha_kg_k$
如何确定系数$\alpha_k$和正交基$g_k$？

1) 求$\alpha_k$
显然$x_{m+1}=x^{*}$，那么$r_{m+1}=b-Ax_{m+1}=b-Ax^{*}=0$
$\begin{aligned} r_{m+1} & =b-Ax_{m+1}\\ & =b-A(x_m+\alpha_mg_m) \\ &=r_{m}-\alpha_mAg_m \\ &=b-Ax_{m}-\alpha_mAg_m \\ &=b-A(x_{m-1}+\alpha_{m-1}g_{m-1})-\alpha_mAg_m \\ & =r_{m-1}-\alpha_{m-1}Ag_{m-1}-\alpha_mAg_m  \\ &=\cdots \\ &= r_k - \alpha_{k}Ag_{k}- \alpha_{k+1}Ag_{k+1} - \cdots -\alpha_mAg_m \\ &=0 \end{aligned}$
则$g_k^Tr_k = \alpha_{k}g_k^TAg_{k}- \alpha_{k+1}g_k^TAg_{k+1} - \cdots -\alpha_mg_k^TAg_m=\alpha_{k}g_k^TAg_{k}$
$\therefore \alpha_{k}=\frac{g_k^Tr_k}{g_k^TAg_{k}}$
2) 求$g_k$
`Gram-shcmit`
    1) 令$g_0=r_0$
    2) 假设已求得$g_0,g_1,\cdots,g_{k-1}$，有$span\{g_0,g_1,\cdots,g_{k-1}\}=span\{r_0,Ar_0,\cdots,A^{k-1}r_0\}$
    3) 对于$r_k$有：
    $$\begin{aligned}r_k&=b-Ax_k \\ &=b-A(x_{k-1}+\alpha_{k-1}g_{k-1}) \\&=r_{k-1}-\alpha_{k-1}Ag_{k-1}\\ &= \cdots \\&=r_0 - \sum_{j=0}^{k-1}\alpha_{i}Ag_{i} \in \Kappa_{k+1}(A,r_0) \end{aligned} $$
    而$g_k \in \Kappa_{k+1}(A,r_0)$，故$r_k=\beta_0g_0+\beta_1g_1+\cdots+\beta_kg_k$
    $\therefore g_i^TAr_k=\beta_ig_i^TAg_i$，$i=0,1,\cdots, k$
    $\therefore g_k$可求

收敛速度大于最速下降法
## A非对称
- 直观法
将$Ax=b$转化为$A^TAx=b$
收敛速度远小于$A$正定时收敛速度
- 广义最小残量法
前面在`Arnoldi`分解我们得到$\Kappa_k(A,r_0)$的标准正交基$v_1,v_2,\cdots,v_k$
令$V_k=(v_1,v_2,\cdots,v_k)$，则$AV_k=V_kH_k+h_{k+1,k}v_{k+1}e_k^T=V_{k+1}\hat{H}_k$
$b-Ax_k=b-A(x_0+V_ky)=r_0-V_{k+1}\hat{H}_ky=V_{k+1}(\left\|r_0\right\|e_1-\hat{H}_ky)$
$\therefore \left\|b-Ax_k\right\|_2= \left\|\left\|r_0\right\|e_1-\hat{H}_ky\right\|_2$
$y=\arg \min_y\left\|\left\|r_0\right\|e_1-\hat{H}_ky\right\|_2$
求解上述式子：
    1) 最小二乘法
    2) Givens旋转变换，对$\hat{H}_k进行QR分解$


# Convex Optimization
## 次梯度
$g^T$ 次梯度
1) 向量：$f(y) \ge f(x) + g^T(y-x)$
2) 矩阵：$f(Y) \ge f(X) + trace(g^T(Y-X))$

例题：求和范数的次梯度$\partial \left\|X\right\|_{*}$
$f(X) \ge f(0) + trace(g^TX) \Rightarrow \sum \sigma_i \ge trace(g^TX)$
$tr(\Sigma_1) \ge tr(g^TU_1\Sigma_1V_1^T) \Rightarrow tr(V_1\Sigma_1V_1^T) \ge tr(g^TU_1\Sigma_1V_1^T)$
$\therefore \partial \left\|X\right\|_{*} =\{U_1V_1^T+Y|U_1^TY=0,YV_1=0,\left\|X\right\|_2 \le 1\}$
# Solving Characteristic Value
