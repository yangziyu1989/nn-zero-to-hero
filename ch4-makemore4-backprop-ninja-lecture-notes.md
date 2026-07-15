# Chapter 4 讲义 · makemore #4 — Becoming a Backprop Ninja

> 视频:Zero to Hero #5 "Building makemore Part 4: Becoming a Backprop Ninja"(~2h)
> 对应 study guide 的 3 个会话。本集形态特殊:一整个大 exercise——把 #3 的两层网络 + BN 的**每一个中间张量**手写反向,与 autograd 对答案。
> 一句话总结:**把 micrograd 的标量链式法则升级到张量:形状对齐定公式,广播与求和互为对偶,多路径依然累加。** 这是 MATLAB → PyTorch 的最后一块地基。

---

## 全局图景:为什么要受这个苦

Karpathy 的檄文《Backprop is a leaky abstraction》:`loss.backward()` 是抽象,但会漏——梯度消失、饱和、BN 的诡异行为、为什么某层学不动,不懂反向的人只能对着黑盒烧香。本集之后,autograd 对你从"必需品"降级为"效率工具"。

工作方式:前向被拆成 ~20 个原子步骤,每个中间张量都有名字;你为每个张量手写 `d<name>`,用 `cmp()` 对照 `tensor.grad` 验收(exact = 逐位相同)。

**三条张量反向心法**(先立后破,全集反复兑现):

1. **形状律**:`dX` 的形状必须与 `X` 完全相同——梯度就是"每个元素动一动,loss 动多少",一一对应。
2. **对偶律**:前向**广播**(复制)⇄ 反向**求和**;前向**求和** ⇄ 反向**广播**。复制到多处 = 多条路径 = 梯度相加(micrograd `+=` 的张量版)。
3. **累加律**:一个张量在前向被用了几次,反向就收几份梯度,全部相加。

---

## 会话 1 · 损失端反向:cross-entropy 链条

前向的分解(#2 手写 softmax 的完全体,含数值稳定的减 max):

```python
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes          # 防上溢
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

### 1.1 逐张量反向(形状律开路)

**dlogprobs**:loss 只挑了每行的正确列取负均值 → 被挑中的位置梯度 −1/n,其余为 0:

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
```

**dprobs**:log 的本地导数 1/x,链上:

```python
dprobs = (1.0/probs) * dlogprobs
```

注意读出直觉:probs 越小(答对的概率越低),梯度被放得越大——**难例自动获得大梯度**。

**dcounts_sum_inv**(对偶律第一次登场):前向 `probs = counts * counts_sum_inv` 中,(n,1) 的 counts_sum_inv 被**广播**成 (n,27) 参与乘法 → 反向必须把 27 列的贡献**求和**收回:

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)   # 形状回到 (n,1) ✅
```

**dcounts**(累加律第一次登场):counts 在前向被用了**两次**——直接乘进 probs,又经 counts_sum 间接参与 → 两份梯度相加:

```python
dcounts = counts_sum_inv * dprobs            # 路径 1
dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum   # 路径 2: sum 的反向 = 广播
```

**dnorm_logits**:exp 的导数是自己:`dnorm_logits = counts * dnorm_logits 链 = counts * dcounts`。

**dlogit_maxes**:算出来 ≈ **0**——这不是巧合:减去每行 max 根本不改变 softmax 的输出(平移不变性,#2 讲过),所以 loss 对它不敏感。手推能亲眼看见这个不变性以梯度=0 的形式显形。

**dlogits**:两条路径(直接项 + 经 logit_maxes 的 one-hot 回传项)相加。

### 1.2 ★ Exercise 2:合并推导,一步到位(课前问题 2)

把 softmax+CE 当一个整体解析求导,结果干净得过分:

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
# 即: dlogits = (P − Y_onehot) / n
```

**为什么这么干净**:softmax 的雅可比与 CE 的导数相乘时大量项对消(亲手推一遍,★ 隔日闭卷任务)。

**物理图像**(Karpathy 花了大篇幅,值得内化):每行梯度 = 预测分布 P 减真实 one-hot。错误类别处梯度为 +P(把 logit **往下拽**,力度=分给它的错误概率),正确类别处为 P−1<0(**往上拉**)。每行梯度之和恰为 0——拉力与拽力平衡,像一组绳索张力。预测完美时 P=Y,梯度处处为 0,绳子松弛。

这也解释了 #2 的伏笔:为什么生产中用 `F.cross_entropy`——前向少算中间量,反向就这三行,又快又稳。

### ⏹ 会话 1 收工自测

- [ ] cross-entropy 链条全部 cmp exact;Exercise 2 三行版与逐步版一致。
- [ ] 口头回答:三条心法?dlogit_maxes 为什么是 0?p−y 的拉/拽图像?

---

## 会话 2 · 线性层 + tanh + embedding 反向

### 2.1 线性层:形状对齐法(课前问题 1)

前向 `logits = h @ W2 + b2`,h:(n,64),W2:(64,27)。反向公式与其死记,不如**用形状唯一确定**:

- `dh` 须是 (n,64);手头有 dlogits (n,27) 和 W2 (64,27) → 唯一拼法 `dlogits @ W2.T`;
- `dW2` 须是 (64,27);手头有 h (n,64) 和 dlogits (n,27) → 唯一拼法 `h.T @ dlogits`;
- `db2` 须是 (27,);b2 前向被广播到每行 → 对偶律,批维求和 `dlogits.sum(0)`。

```python
dh  = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
```

**为什么出现转置**:标量看,`dW2[i,j] = Σ_batch h[·,i] · dlogits[·,j]`——每个权重的梯度是"它的输入 × 它输出处的梯度"在 batch 上求和;写成矩阵形式自然要求 h 转置。小抄:**"输入的转置在梯度公式里出现"= micrograd 乘法节点"交换"规则的矩阵版。**

**batch 维为什么是求和不是平均**(课前问题 3):W2 在前向被 batch 里**每个样本各用了一次**(参数共享=多路径)→ 累加律,求和。而"平均"其实已经发生了——藏在最上游 `loss = mean(...)` 的 1/n 里,顺着 dlogits 传下来。若 loss 用 sum 而非 mean,这里就真的只有求和、梯度整体大 n 倍(lr 要相应除 n)——所以说**求和还是平均由 loss 的定义决定,层的反向永远是求和**。

### 2.2 tanh 与 BN 的 γ/β

```python
dhpreact = (1.0 - h**2) * dh          # micrograd 老朋友, 逐元素
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # γ 被广播 → 求和, keepdim 保形状
dbnbias = dhpreact.sum(0, keepdim=True)
```

注意 keepdim:γ/β 形状是 (1,64),求和后必须 keepdim 才能 cmp exact——**形状律连"到底是 (64,) 还是 (1,64)"都管**。

### 2.3 embedding:花式索引的反向(课前问题 2)

前向 `emb = C[Xb]`(查表)→ 反向:demb 里每个位置的梯度要**加回**它当初查的那一行:

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        dC[Xb[k, j]] += demb[k, j]     # += ! 同一行被索引多次, 梯度汇聚
# 向量化等价: dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))
```

同一个字符(比如 'a')在 batch 里出现 500 次,C['a'] 那一行就收 500 份梯度之和——**这正是 micrograd `+=` bug 的工业级现场**:如果这里写 `=`,高频字符的 embedding 学习会系统性出错,而 loss 看起来还在降。

### ⏹ 会话 2 收工自测

- [ ] 线性层/tanh/embedding 全部 cmp exact。
- [ ] 口头回答:形状对齐法推 dW;batch 维求和的理由及"何时变平均";embedding 行的梯度汇聚。

---

## 会话 3 · BatchNorm 反向 + 整合

### 3.1 为什么 BN 反向难(课前问题 1)

其他层是"一个输入 → 一个输出"的干净映射;BN 里**每个输出 xhat[i] 依赖 batch 里全部输入**——因为均值 μ 和方差 σ² 都是所有样本的函数。于是 ∂loss/∂x[i] 有三条路径:

```
x[i] → xhat[i]         (直接路径)
x[i] → μ → 所有 xhat    (均值路径)
x[i] → σ² → 所有 xhat   (方差路径)
```

处理办法没有新数学:把前向拆成原子步(bnmeani → bndiff → bndiff2 → bnvar → bnvar_inv → bnraw),每步用三条心法机械反向,最后累加三条路径。注意视频用 **Bessel 校正**(除 n−1)算方差——小 batch 下无偏;手推时除数写错是 cmp 不 exact 的头号原因。

### 3.2 解析合并:一行公式及其解剖(课前问题 2)

逐步反向验证通过后,像 Exercise 2 一样在纸上合并(★ 值得完整推一遍),得到:

```python
dhprebn = bngain * bnvar_inv / n * (
            n * dhpreact                                # ① 直接项
            - dhpreact.sum(0)                           # ② 减去梯度的均值
            - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)  # ③ 减去沿 bnraw 方向的分量
          )
```

解剖(对应课前问题"哪些项对应去均值、哪些对应去相关"):

- **① 直接项**:若 μ、σ 是常数,梯度本该就是 `bngain·bnvar_inv·dhpreact`;
- **② 去均值项**:来自均值路径——把上游梯度的 batch 均值扣掉。含义:BN 输出对"整个 batch 一起平移"不敏感(平移会被减均值吃掉),所以梯度中"共同平移"的分量必须清零;
- **③ 去相关/去尺度项**:来自方差路径——把梯度中与 bnraw 同方向的分量按其投影扣除。含义:BN 输出对"整个 batch 一起缩放"也不敏感,梯度里"沿数据本身方向"的成分要移除。

一句话:**前向除掉了平移和尺度两个自由度,反向就必须把梯度投影到与这两个自由度正交的子空间。** 这个"归一化 ⇄ 梯度投影"的观点在读任何 Norm 类论文时都用得上。

### 3.3 Exercise 4:全手动训练循环

把所有手写梯度串起来,替换 `loss.backward()`,完整训练 MLP——loss 曲线与 autograd 版重合。至此你写过一个不依赖 autograd 的完整神经网络训练器。

### 3.4 一句话能力总结(课前问题 3,自己写,这里给参照)

参照答案:"**任何由可微算子组成的计算图,给我前向代码,我就能手写出它的反向,并且知道每一项梯度的物理含义。**" 收工时和你自己的版本对照修订。

### ⏹ 会话 3 收工自测

- [ ] 全部 exercises 完成,cmp 全 exact;手动梯度版训练循环跑通。
- [ ] ★ 隔日闭卷:手推 softmax+CE 合并梯度(p−y)/n,含"为什么行和为 0"。

---

## Chapter 4 复习卡

1. **三条心法**:形状律(dX 与 X 同形)、对偶律(广播⇄求和)、累加律(多用多收)。90% 的张量反向由这三条机械生成。
2. **两个必背结果**:softmax+CE → `(P − Y)/n`;线性层 → `dW = X.T @ dY, dX = dY @ W.T, db = dY.sum(0)`。
3. **两个梯度=0 的"不变性显形"**:dlogit_maxes≈0(softmax 平移不变);BN 反向扣除平移/缩放分量(归一化的不变性)。看到梯度恒 0,先想"前向是不是对这个量不敏感"。
4. **broadcasting 反向补偿是 MATLAB→PyTorch 头号易错点**(总纲原话):前向哪里形状被拉伸了,反向就在哪里 sum 回去,keepdim 跟着前向的形状走。
5. embedding/共享参数的梯度**永远是汇聚累加**——micrograd 的 `+=` 从 Chapter 1 一路追到这里,完成闭环。

**易错点清单**:
- 反向漏掉一条使用路径(counts、logits 都被用两次)
- 广播维忘 sum、或 sum 后忘 keepdim(形状差一个维度,cmp approximate 但不 exact)
- 方差的 n vs n−1(Bessel)不一致
- dC 用 `=` 覆盖而非 `+=` 累加
- 转置位置靠背公式而不是形状对齐(换个形状就错)

---

## 与后续章节的钩子

- **→ #5/GPT**:此后 `loss.backward()` 放心用——但每当出现"某层学不动/梯度爆炸",你现在有能力直接拆开看是哪一项出了问题。
- **→ GPT 视频**:残差连接为什么让梯度畅通,你已经有答案——`+` 节点把梯度原样分发,恒等支路 = 无衰减高速公路(micrograd 第一课 + 本集的张量版)。
- **→ 6.S184 / DDPM 手推**:ELBO→简化目标的推导用的是同一套"链式法则 + 对不变量敏感度为零"的肌肉。
