<!-- $theme: gaia -->

# 機械学習勉強会
##### 第５回
##### SRA 鈴木真吾

----
今回の内容

詳解ディープラーニング 4.5 高度なテクニック

----

データ・パラメタは学習の結果に影響を与える。
ではどのようなデータ・パラメタを取るのがよいか？

- データセット
- 重み
- 学習率
- 学習回数

----

- **データセット**
- 重み
- 学習率
- 学習回数

-----
### データセットに関するテクニック
- 正規化
- Batch Normalization (後半で説明)
----
## 正規化

```python
X = X - X.mean(axis=1).reshape(len(X), 1)
```

- 値域が一定の範囲に収まるようにする
  - 任意のデータで同じアプローチを取りたい
  - 処理しやすいデータにする 
    - 平均は$0$になるようにする
- 特に次の場合は白色化という
  - 平均が$0$
  - 分散が$1.0$

----

- データセット
- **重み**
- 学習率
- 学習回数

----
### どういう重みだとうれしいか？
- 前提としてデータセットに偏りがない
  - → 重みにも偏りは少ないはず
- すべて $0$ にしたらどうなるか？
  - → 誤差逆伝播の際に勾配の値も同じになって学習が進まなくなってしまう
- 小さい標準偏差で分布させれば？
  - → 小さいとやはり学習が進まない
----
#### 結論
  - 標準偏差は$\sigma=1.0$で分布させて、適当な係数$a$をかけて初期化しよう 
  - python コードとしては
   ```python
a * np.random.normal(size=shape)
   ```
----
活性化される前の値=次の層の入力となる

$$
\begin{aligned}
p_j = \sum^{n}_{i=1}w_{ij}x_i & & (4.2)
\end{aligned}
$$

----

$$
\begin{aligned}
Var\lbrack p_j\rbrack=&Var\begin{bmatrix}\sum^n_{i=1}w_{ij}x_i\end{bmatrix} & & (4.43) \\
=& \sum^n_{i=1}Var \begin{bmatrix}w_{ij}x_i\end{bmatrix} && (4.44) \\
=& \sum^n_{i=1} \begin{Bmatrix}
&(E[w_{ij}])^2Var[x_i]\\
+&(E[x_{i}])^2Var[w_{ij}]\\
+&Var[w_{ij}]Var[x_i]\end{Bmatrix} && (4.45)
\end{aligned}
$$


----
#### 補足
- 確率変数の期待値の積　（４．４４から4.45の変形に使用)
$$
\begin{aligned}
E[XY] =& \sum_{x,y}P(X=x,Y=y)xy \\
=& \sum_x\sum_yP(X=x)P(Y=y)xy \\
=& \sum_xP(X=x)x\sum_yP(Y=y)xy \\
=& E[X]E[Y]
\end{aligned}
$$

----
#### 補足
- 確率変数の分散
$$
\begin{aligned}
Var[X] =& E[X^2]-E[X]^2
\end{aligned}
$$
- 確率変数の分散の積
$$
\begin{aligned}
Var[XY] =& E[X^2Y^2] - E[X]^2E[Y]^2 \\
=& E[X^2]E[Y^2] - E[X]^2E[Y]^2 \\
=& (Var[X]+E[X]^2)(Var[Y]+E[Y]^2) \\
 & - E[X]^2E[Y]^2\\
=& Var[X]Var[Y] \\
 & + Var[X]E[Y]^2 + Var[Y]E[X]^2  
\end{aligned}
$$

----

(4.45) は
- $[x_i] = 0$とデータを正規化している
- $E[w_{ij}] = 0$ と仮定する
すると次のようになる
$$
\begin{aligned}
Var\lbrack p_j\rbrack&=&\sum^n_{i=1}Var[w_{ij}]Var[x_i] && (4.46) \\
&=&nVar[w_{ij}]Var[x_i] && (4.47) \\
\end{aligned}
$$

----
$$
\begin{aligned}
p_j =& \sum^{n}_{i=1}w_{ij}x_i & & (4.2) \\
Var[p_j\rbrack=&nVar[w_{ij}]Var[x_i] && (4.47) \\
\end{aligned}
$$

----

$p$の分散を$x$の分散に合わせたい場合、

$Var[p_j]=Var[x_i]$とすると、
$Var[w_{ij}]=\frac{1}{n}$、$Var[aX]=a^2Var[X]$なので

$$
\begin{aligned}
a&=& \sqrt{\frac{1}{n}} && (4.48) \\
\end{aligned}
$$
とすればよい

----

Pythonの式だと
```python
np.sqrt(1.0 / n) * np.random.normal(size=shape)
```

$Var[P_{ij}]$を決めるまでにあった仮定のとりかたによって、初期化手法はいくつか考えられる

----

## LeCun et al. 1988
- 入力数の平方根でスケーリングした一様分布による初期化

```python
np.random.uniform(low=-np.sqrt(1.0 / n),
                  high=np.sqrt(1.0 / n),
                  size=shape)
```

----

## Glorot and Bengio 2010
- fan_in  + fan_out でスケーリングした一様分布による初期化

```python
np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                  high=np.sqrt(6.0 / (n_in + n_out)),
                  size=shape)
```
----

## He et al. 2015
- `ReLU`を使った初期化

```python
np.sqrt(2.0 / n) * np.random.normal(size=shape)
```
----

- データセット
- 重み
- **学習率**
- 学習回数

----

### 学習率に関するテクニック
- モメンタム
- Nesterov モメンタム
- Adagrad
- Adadelta
- Adam

----
#### モメンタム

これまでは学習率は定数だったが、効率的に学習をすすめるには
  - 最初は大きく学習をすすめる
  - 徐々に学習率を少なくする

*モメンタム*の考え方
  - 学習率は一定
  - モメンタム項という調整用の項を追加して上記を表現する
----

- モデルのパラメータ$\theta$
- $E$の$\theta$に対する勾配を$\nabla_{\theta}E$
- ステップを$t$

とした時のパラーメタ更新式は次のようになる

$$
\begin{aligned}
\Delta\theta^{(t)}=&
-\eta\nabla_{\theta}E(\theta)+\color{red}{\gamma\Delta\theta^{(t-1)}}
&& (4.49)
\end{aligned}
$$
- $\gamma\Delta\theta^{(t-1)}$がモメンタム項
- 係数$\gamma(<1.0)$は通常$0.5$から$0.9$に設定する

---------

TensorFlow
```python
def training(loss):
    optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    train_step = optimizer.minimize(loss)
    return train_step
```

Keras
```python
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

```
---------

#### Nesterovモメンタム

- $(4.49)$の変形
$$
\begin{aligned}
\upsilon^{(t)}=&
-\eta\nabla_{\theta}E(\theta)+\gamma\Delta\theta^{(t-1)}
&& (4.51) \\
\theta^{(t)}=&
\theta^{(t-1)}-\upsilon^{(t)}
&& (4.52)
\end{aligned}
$$

- $\Delta\theta^{(t)}=\theta^{(t)}-\theta^{(t-1)}=\upsilon^{(t)}$ と変形して
$$
\begin{aligned}
\upsilon^{(t)}=&
-\eta\nabla_{\theta}E(\theta\color{red}{+\gamma\upsilon^{(t-1)}})+\gamma\Delta\theta^{(t-1)}
&& (4.53) \\
\theta^{(t)}=&
\theta^{(t-1)}-\upsilon^{(t)}
&& (4.54)
\end{aligned}
$$

-------

TensorFlow
```python
optimizer = tf.train.MomentumOptimizer(
    0.01, 0.9, use_nesterov=True)
```

Keras
```python
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True
```

--------
#### Adagrad

効率的に学習をすすめるには
- 最初は大きく学習をすすめる
- 徐々に学習率を少なくする
Adagrad(adaptive gradient algorithm)の考え方
- 学習率の値そのものを更新する。

----

Adagraは次の式

$$
\begin{aligned}
\theta^{(t)}_i=&
\theta^{(t-1)}_i - \frac{\eta}{\sqrt{G^{(t)}_{ii}+\epsilon}}g_i^{(t)}
&& (4.56)
\end{aligned}
$$

ただし、

$$
\begin{aligned}
g_i&\colon=&\nabla_{\theta}E(\theta_i)
&& (4.55) \\
G^{(t)}_{ii}&=&
\sum^t_{\tau=0}g^{(\tau)}_{i}\centerdot g^{(\tau)}_{i}
&& (4.57)
\end{aligned}
$$
- $\epsilon$は$1.0\times 10^{-6}\sim 1.0\times 10^{-8}$程度の微小な項
  - 0除算の回避用途 

----

$G$は対角行列なので、(4.56)は要素積に置き換えられる

$$
\begin{aligned}
\theta^{(t)}_i=&
\theta^{(t-1)}_i - \frac{\eta}{\sqrt{G^{(t)}_{ii}+\epsilon}}\odot g_i^{(t)}
&& (4.56)
\end{aligned}
$$

- 通常の勾配降下法では$\theta^{(t)}_i=\theta^{(t-1)}_i - \eta g_i^{(t)}$
- $G_{ii}$は、 $t$ までの勾配の２乗和
  - 直感的にはこれまでのステップで小さかった勾配の成分が次のステップでは大きくなるように更新される

----

#### Adadelta

Adagrad の問題点
- $G^{(t)}$ は勾配の2乗の累積和$=$単調増加
  - 学習のステップが進む毎に勾配にかかる係数が急激に小さくなり、学習がすすまなくなる
Adadelta では
- → ステップ $0$ からの全ての和でなく、定数$w$ のステップ文の和に制限する
 - 実装としては非効率なので、実際には減衰平均する

----

「勾配の2乗」$(=g_t\odot g_t)$の移動平均$E[g^2_t]$は

$$
\begin{aligned}
E[g^2]_t&=&
\rho E[g^2]_{t-1}+(1-\rho)g^2_t
&& (4.59)
\end{aligned}
$$

(以降では$g^{(t)}$は$g_t$と表記)

----

Adagradの式
$$
\begin{aligned}
\theta_{t+1}=&
\theta_{t} - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
&& (4.60)
\end{aligned}$$

$このG_t$を$E[g^2_t]$で置は換える

$$
\begin{aligned}
\theta_{t+1}=&
\theta_{t} - \frac{\eta}{\sqrt{E[g^2]_{t} + \epsilon}} g_t
&& (4.61)
\end{aligned}
$$

さらに$\sqrt{E[g^2]_{t}}$は$RMS[g]_t$に置き換えられて

$$
\begin{aligned}
\theta_{t+1}=&
\theta_{t} - \frac{\eta}{RMS[g]_{t}} g_t
&& (4.62)
\end{aligned}
$$


----
#### 補足

$RMS$は2乗平均平方根 (root mean square)
$$
\begin{aligned}
RMS[x]&=&
\sqrt{\frac{1}{N}\sum^N_{i=1}x_i^2}
&& (C1)
\end{aligned}
$$

----

さらに式変形すると(4.63)の式が得られる
$$
\begin{aligned}
\Delta\theta_{t}=&
- \frac{\eta}{RMS[g]_{t}} g_t
&& (4.63)
\end{aligned}
$$

$\eta$に何が入るか考えた時、
- 左辺・右辺の「単位」は揃っているはず
- $\Delta \theta_t$は$t-1$までの$\Delta \theta_t$ のRMSから近似できる
と考えて、Adadeltaの式が得られる

$$
\begin{aligned}
\Delta\theta_t
=&
-\frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t}g_t
&& (4.66)
\end{aligned}
$$

となって学習率$\eta$が自動的に計算できることになる

----

- 元の論文読むと

> ∆x t for the current
  time step is not known, so we assume the curvature is locally
  smooth and approximate ∆x t by compute the exponentially
  decaying RMS over a window of size w of previous ∆x

この$\Delta x$が $\Delta \theta$のこと

----
#### RMSprop

RMSprop は Adadelta と同様に、Adagrad の学習率の急激な減少を解決する手法

$$
\begin{aligned}
E[g~2]_t
&=&
0.9E[g~2]_{t-1}+0.1g~2_t
&& (4.67)
\end{aligned}
$$

$$
\begin{aligned}
\theta_{t+1}&=&
\theta_{t} - \frac{\eta}{\sqrt{E[g~2]_{t}+\epsilon}} g_t
&& (4.68)
\end{aligned}
$$

---
#### Adam(adaptive moment estimation)

つぎの２つをパラメータの更新式に使う方式
- 勾配の２乗の移動平均$v_t:=E[g^2]_t$の減衰平均
- 勾配の単純な移動平均$m_t:=E[g]_t$の減衰平均

$$
\begin{aligned}
m_t =& \beta_1m_{t-1}+(1-\beta_1)g_t
&& (4.69) \\
v_t =& \beta_2v_{t-1}+(1-\beta_2)g^2_t
&& (4.70)
\end{aligned}
$$

- $\beta_1,\beta_2\in\lbrack 0,1)$はハイパーパラメタ
  - 移動平均の減衰率を調整

----
$v_t$,$m_t$は真のモーメントから偏りがあるので、この偏りを$0$にした推定値$\hat{v_t}$,$\hat{m_t}$を求めたい。

$$
\begin{aligned}
v_t =& \beta_2v_{t-1}+(1-\beta_2)g^2_t
&& (4.70)
\end{aligned}
$$

について、$v_0=0$で初期化した場合、$v_t$は

$$
\begin{aligned}
v_t =& (1-\beta_2)\sum^t_{i=1}\beta^{t-i}_2 \centerdot g^2_i
&& (4.71)
\end{aligned}
$$

- $(1-\beta_2)^n\sum_{i=n}^t\beta_2^{t-i} \centerdot g^2_i$のような項もあるんじゃないかと思うが、$(1-\beta_2)^n\ll 1$として無視されていると思われる 

----
$$
\begin{aligned}
v_t =& (1-\beta_2)\sum^t_{i=1}\beta^{t-i}_2 \centerdot g^2_i
&& (4.71)
\end{aligned}
$$

ここから2次モーメントの移動平均$E[v_t]$と真の2次モーメント$E[g^2_t]$の関係を求めると

$$
\begin{aligned}
E[v_t] =&
E\begin{bmatrix}(1-\beta_2)\sum^t_{i=1}\beta^{t-i}_2 \centerdot g^2_i\end{bmatrix}
&& (4.72) \\
=&
E[g^2_t]\centerdot(1-\beta_2)\sum^t_{i=1}\beta^{t-i}_2 + \zeta
&& (4.73) \\
=&
E[g^2_t]\centerdot (1-\beta^t_2) + \zeta
&& (4.74)
\end{aligned}
$$

----
#### 補足

(4.73)から(4.47)の変形
$$
\begin{aligned}
&
(1-\beta)\sum^t_{i=1}\beta^{t-i}
\\
=&
(1-\beta)(\beta^{t-1} + \beta^{t-2} + \cdots + \beta + 1)
\\
=& ~~~~~~~~~~~~ \beta^{t-1} + \beta^{t-2} + \cdots + \beta  + 1 \\
 & - \beta^t - \beta^{t-1} - \beta^{t-2} - \cdots - \beta\\
=& - \beta^t + 1
\end{aligned}
$$

----

$\zeta=0$と近似できるようにハイパーパラメタの値を設定すると次のように推定できる。

$$
\begin{aligned}
\hat{v_t} =& \frac{v_t}{1-\beta^t_2}
&& (4.75)
\end{aligned}
$$

$m_t$についても同様

$$
\begin{aligned}
\hat{m_t} =& \frac{m_t}{1-\beta^t_1}
&& (4.76)
\end{aligned}
$$

以上から、パラメタの更新式は

$$
\begin{aligned}
\theta_t =& \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}
&& (4.77)
\end{aligned}
$$

----
- データセット
- 重み
- 学習率
- **学習回数**

----
### Early Stopping

学習回数は
- 多いほど訓練データへの誤差は小さくなる
- 多すぎるとオーバーフィッティングが発生する
→ Early Stopping

手法としては「前のエポックの時と比べ誤差が増えたら学習を打ち切る」

----

##### Early Stopping の擬似コード
```python
for epoch in range(epochs):
    loss = model.train()['loss']

    if early_stopping(loss):
        break
```

----
- **データセット**
- 重み
- 学習率
- 学習回数

----
### Batch Normalization

- 前処理としての正規化
  - 学習の際にはネットワーク内部で分散がかたよる
- Batch Normalization
  - 正規化をミニバッチに対しても行う手法

----
ミニバッチ$\mathfrak{B}=\{ x_1, x_2, ..., x_m \}$に対して

$$
\begin{aligned}
\mu_{\mathfrak{B}}=&
\frac{1}{m}\sum^m_{i=1}x_i^2
&& (4.79) \\
\sigma^2_{\mathfrak{B}}=&
\frac{1}{m}\sum^m_{i=1}(x_i-\mu_{\mathfrak{B}})^2
&& (4.80) \\
\end{aligned}
$$

----

$$
\begin{aligned}
\hat{x_i} =& \frac{x_i-\mu_{\mathfrak{B}}}{\sqrt{\sigma^2_{\mathfrak{B}} + \epsilon}}
&& (4.81) \\
y_i =& \gamma\hat{x_i}+\beta && (4.82)
\end{aligned}
$$

$\{y_1,y_2,...,y_m\}$ がBatch Normalization の出力

----

誤差関数$E$に対して、$\gamma$、$\beta$、$x_i$の勾配を求める

$$
\begin{aligned}
\frac{\partial E}{\partial \gamma}
&= \sum_{i=1}^m \frac{\partial E}{\partial y_i} \frac{\partial y_i}{\partial \gamma}
&& (4.83) \\
&= \sum_{i=1}^m \frac{\partial E}{\partial y_i}\hat{x_i} && (4.84)
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial E}{\partial \beta}
&= \sum_{i=1}^m \frac{\partial E}{\partial y_i} \frac{\partial y_i}{\partial \beta}
&& (4.83) \\
&= \sum_{i=1}^m \frac{\partial E}{\partial y_i} 
&& (4.85)
\end{aligned}
$$

----

$$
\begin{aligned}
\frac{\partial E}{\partial x_i}
=& 
\frac{\partial E}{\partial \hat{x_i}} \frac{\partial \hat{x_i}}{\partial x_i}
+ \frac{\partial E}{\partial \sigma^2_{\mathfrak{B}}} \frac{\partial \sigma^2_{\mathfrak{B}}}{\partial x_i}
+ \frac{\partial E}{\partial \mu_{\mathfrak{B}}} \frac{\partial \mu_{\mathfrak{B}}}{\partial x_i}
\\
&(4.87) \\
\\
=& 
\frac{\partial E}{\partial \hat{x_i}}
\frac{1}{\sqrt{\sigma^2_{\mathfrak{B}} + \epsilon}}
+ 
\frac{\partial E}{\partial \sigma^2_{\mathfrak{B}}}
\frac{2(x_i-\mu_{\mathfrak{B}})}{m}
+ \frac{\partial E}{\partial \mu_{\mathfrak{B}}} \frac{1}{m}
\\
& (4.88)
\end{aligned}
$$

----

$\frac{\partial E}{\partial y_i}$は既知。他の項は

$$
\begin{aligned}
\frac{\partial E}{\partial \hat{x_i}}
=& \frac{\partial E}{\partial y_i}
   \frac{\partial y_i}{\partial \hat{x_i}}
&& (4.89) \\
=& \frac{\partial E}{\partial y_i}
\centerdot \gamma
&& (4.90)
\end{aligned}
$$
$$
\begin{aligned}
\frac{\partial E}{\partial \sigma^2_{\mathfrak{B}}}
=& \sum_{i=1}^m 
   \frac{\partial E}{\partial \hat{x_i}}
   \frac{\partial \hat{x_i}}{\sigma^2_{\mathfrak{B}}}
   \\
& (4.91) \\
=& \sum_{i=1}^m 
   \frac{\partial E}{\partial \hat{x_i}}
\centerdot
    (x_i - \mu_{\mathfrak{B}})
\centerdot
    \frac{-1}{2}
    (\sigma^2_{\mathfrak{B}}+\epsilon)^{-\frac{3}{2}} \\
& (4.92)
\end{aligned}
$$

----

$$
\begin{aligned}
\frac{\partial E}{\partial \mu_{\mathfrak{B}}}
=& \sum_{i=1}^m 
   \frac{\partial E}{\partial \hat{x_i}}
   \frac{\partial \hat{x_i}}
        {\mu_{\mathfrak{B}}}
 + 
   \frac{\partial E}
        {\partial \sigma^2_{\mathfrak{B}}}
   \frac{\partial \sigma^2_{\mathfrak{B}}}
        {\mu_{\mathfrak{B}}}
& (4.93) \\
=& \sum_{i=1}^m 
   \frac{\partial E}{\partial \hat{x_i}}
   \frac{-1}
        {\sqrt{\sigma^2_{\mathfrak{B}}+\epsilon}}
\\
 &+ \sum_{i=1}^m
   \frac{\partial E}
        {\partial \sigma^2_{\mathfrak{B}}}
   \frac{-2(x_i-\mu_{\mathfrak{B}})}
        {m}
& (4.94)
\end{aligned}
$$

前の層の勾配を現在の層の情報だけで求められるので、誤差逆伝播法がつかえる。

----

これまでの層の活性化は

$$
\begin{aligned}
\bold{h} &= f(W\bold{x}+\bold{b})  & (4.95)
\end{aligned}
$$

Batch Normalization では次のようになりバイアス項がなくなる

$$
\begin{aligned}
\bold{h} &= f(BN_{\gamma,\beta}(W\bold{x}))
& (4.96)
\end{aligned}
$$

----

#### まとめ

- 初期化方法はいろいろあり、どれが最適とは必ずはいえない
  - とはいえAdadelta とBatch Normalization は良いらしい
- 問題によって最適な初期化を探すことになりそう