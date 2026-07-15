# Chapter 2 讲义(上) · makemore #1 — bigram 语言模型

> 视频:Zero to Hero #2 "The spelled-out intro to language modeling: building makemore"(~2h)
> 对应 study guide 的 2 个会话。数据:names.txt(32,033 个英文名字)。
> 一句话总结:**语言模型 = 对"下一个 token 的条件概率分布"建模。** 本视频用同一个 bigram 任务做两遍——先"数数",再"神经网络"——并证明两者殊途同归。这个"计数 ↔ 梯度下降"的等价性是全视频的灵魂。

---

## 全局图景:makemore 系列在做什么

- 任务始终不变:**字符级语言模型**——给定前文字符,预测下一个字符,从而能"生成更多像 names.txt 的名字"(make more)。
- 从 bigram(#1)到 MLP(#2)、BatchNorm(#3)、手写 backprop(#4)、WaveNet(#5),直到 GPT——**变的只是"用多长的上下文、用什么架构去建模这个条件分布"**,任务与 loss(NLL)从头到尾一样。
- bigram 是能work的最小语言模型:只看**前一个**字符。它丢掉的信息 = 前一个字符之前的**全部历史**(会话 1 课前问题 1)。名字"是否已经很长了""开头是什么风格"它一概不知,所以生成质量必然差——但管线(数据 → 概率 → 采样 → 评估)与 GPT 完全同构。

---

## 会话 1 · 计数模型 + 采样 + NLL

### 1.1 数据与 bigram 的提取

```python
words = open('names.txt').read().splitlines()   # 32033 个名字, 长度 2~15
```

关键设计:**特殊 token 标记边界**。名字不是无限流,"从哪开始、到哪结束"本身是要学的分布。视频先用 `<S>`/`<E>` 两个 token,随后简化为一个 `'.'` 通吃(既当开始也当结束),词表 = 26 字母 + `'.'` = **27**。

```python
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0                                   # '.' 排在 0 号
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):          # 滑窗取相邻对
        ...
```

`zip(chs, chs[1:])` 这个惯用法 = 你熟悉的**滑动窗口/一阶差分索引**,后面所有 n-gram 上下文构造都是它的推广。

### 1.2 计数矩阵 N 与概率矩阵 P

```python
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
```

- `N[i, j]` = 字符 i 后面跟字符 j 的次数。用 `plt.imshow` 可视化后能直接读出结构:元音行很亮、`.q` 之后几乎只有 `u` 等。**统计建模第一步永远是看数据**。
- 行归一化得到条件概率:每行是一个 `P(next | current)` 的 categorical 分布。

```python
P = (N + 1).float()                  # +1 = 平滑, 见 1.5
P /= P.sum(1, keepdim=True)          # 每行除以行和 → 行归一化
```

### ⚠ 1.3 broadcasting 深潜:keepdim 陷阱(全视频最重要的工程段落)

Karpathy 在这里刻意停下来讲了十分钟,因为这是**静默出错**的典型:

- `P.sum(1, keepdim=True)` 形状 `(27, 1)`:除法时列向广播,**每行除以自己的行和** ✅
- `P.sum(1)`(不写 keepdim)形状 `(27,)`:广播规则是**右对齐补 1**,变成 `(1, 27)`——除法变成了**每列除以一个数**,即"用第 j 列的行和去除第 j 列" ❌
- 恶劣之处:形状合法、不报错、数值看着也像概率,但**归一化的轴错了**。只有去检查 `P[0].sum()` 是否为 1 才能发现。

对 MATLAB 背景的你:这相当于 `bsxfun` 的扩展方向搞反,但 PyTorch 不会警告。**养成习惯:每写一个带 reduce 的广播算式,停下来自问——对齐后每个维度分别是拉伸谁?归一化轴对不对?** 这个习惯会在 #4(手写 backprop 的广播补偿)直接变现。

另一个细节:`P /= ...`(原地)比 `P = P / ...` 省一份内存——张量大了以后有意义。

### 1.4 采样:从概率表滚骰子(课前问题 2)

```python
g = torch.Generator().manual_seed(2147483647)
ix = 0                                          # 从 '.' 开始
out = []
while True:
    p = P[ix]                                   # 当前行 = 条件分布
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    if ix == 0:                                 # 采到 '.' = 结束
        break
    out.append(itos[ix])
''.join(out)     # 'junide', 'janasah', ...
```

操作分解:**取当前字符对应的行 → 该行是一个 categorical 分布 → `torch.multinomial` 按概率抽一个索引 → 该索引成为新的当前字符 → 循环直到抽到 `'.'`**。这个"查条件分布 → 抽样 → 反馈"循环与 GPT 的自回归采样逐字同构,只是 GPT 的"行"由 Transformer 现算。

生成的名字很难听,但对照实验(把 P 换成均匀分布)生成的更糟——说明模型确实学到了东西。**"比 baseline 好"永远比"绝对好"先说明问题**。

### 1.5 评估:似然 → NLL(课前问题 3)

模型质量的定义:**它给训练数据的概率有多高**(似然)。

$$\text{likelihood} = \prod_{i} P(x_{i+1} \mid x_i)$$

三步变换,每步都有明确理由:

1. **取 log**:上千个 <1 的数连乘会下溢到 0;log 单调,不改变排序,乘法变加法。(你的 dB 直觉:同一招。)
2. **取负**:log-likelihood ≤ 0,越大越好;习惯上 loss 要"越小越好",所以加负号 → NLL。
3. **取平均**:除以 bigram 总数,消除数据集长度影响,得到可跨数据集比较的单个数字。

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1
nll = -log_likelihood / n        # ≈ 2.454
```

**基准线**:瞎猜 = 27 类均匀分布,平均 NLL = −log(1/27) = log 27 ≈ **3.296**。计数模型 2.454 < 3.296,确认有效。这两个数字要记住——下一个视频的 MLP 和第一次训练 GPT 时,你都要先算"瞎猜 loss"来判断初始化是否正常(#3 的核心主题之一)。

**平滑(smoothing)**:若某 bigram 计数为 0(如 "jq"),其概率为 0,一旦测试数据里出现,log 0 = −∞,NLL 爆炸。解法:`P = (N + 1).float()`——所有计数 +1(可调成 +k,k 越大分布越接近均匀)。记住这个操作,会话 2 它会以正则化的身份再次登场。

### ⏹ 会话 1 收工自测

- [ ] 计数版 bigram 能生成名字;NLL ≈ 2.454 复现。
- [ ] 口头回答:bigram 丢了什么信息?采样循环四步?为什么 log、为什么负、为什么平均?瞎猜 NLL 是多少、怎么算的?
- [ ] 说出 keepdim 陷阱错在哪个轴。

---

## 会话 2 · 神经网络版 bigram + 等价性

### 2.1 换个问法:同一分布,改用可微参数表示

计数法直接"查表填概率"。现在改成:**参数矩阵 W 经过一条可微管道产生概率,用梯度下降调 W 使 NLL 最小**。任务、loss、评估全部不变——变的只是"分布从哪来"。

训练集 = 全部 bigram 对(228,146 个样本):

```python
xs, ys = [], []                     # xs: 当前字符索引, ys: 下一个字符索引
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1]); ys.append(stoi[ch2])
xs = torch.tensor(xs); ys = torch.tensor(ys)
```

### 2.2 one-hot 乘矩阵 = 选行(课前问题 1)

整数索引不能直接进网络(它不是量,是类别名),先 one-hot:

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()    # (N, 27), 注意要 .float()
W = torch.randn((27, 27), generator=g, requires_grad=True)
logits = xenc @ W                               # (N, 27)
```

**关键恒等式:one-hot 向量 @ W ≡ 取出 W 的对应行。** 第 i 位为 1 的 one-hot 乘 W,结果就是 `W[i]`。所以这个"单层网络"实际上还是在查表——与计数法的 `N[i]` 结构完全平行。这个恒等式在 #2 会直接升级为 **embedding 查表**(`C[ix]` 就是省掉乘法的 one-hot @ C),现在就把它焊牢。

### 2.3 logits → softmax(课前问题 2)

W 的输出是任意实数,而我们需要"每行非负、和为 1"的概率。管道:

```python
counts = logits.exp()                           # 全部变正 → 解读为"伪计数"
probs = counts / counts.sum(1, keepdim=True)    # 行归一化 → 概率
# 这两行合起来就是 softmax
```

- **为什么叫 logits**:它们被解读为 **log(counts)**——取 exp 恰好还原成(正的)伪计数,再归一化。命名即语义:logits = 对数计数。
- exp 的作用:把 (−∞, +∞) 单调地映到 (0, +∞),保证概率非负且保序。
- 与计数模型逐项对齐:`exp(logits) ↔ N 的行`,`归一化 ↔ P 的行归一化`。**神经网络版从头到尾在模仿计数版,只是"计数"变成了可微参数。**

### 2.4 loss 与训练循环

对每个样本,取模型分给**正确下一字符**的概率,求平均 NLL:

```python
loss = -probs[torch.arange(num), ys].log().mean()
# 花式索引: 第 i 行取第 ys[i] 列 —— "正确答案的概率"
```

完整循环(与 micrograd 四步一模一样,只是张量化了):

```python
for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    W.grad = None            # PyTorch 版 zero_grad
    loss.backward()
    W.data += -50 * W.grad   # lr=50: 这个问题太简单, 大步长也稳
print(loss.item())           # → 收敛到 ≈ 2.48, 逼近计数版的 2.454
```

注:lr=50 看着吓人,但这里 loss 面是一个近乎凸的浅碗(单线性层 + softmax + NLL),大步长无碍。**不要把它当经验带去深网络。**

### 2.5 ★ 等价性:为什么 loss 必然"几乎相同"

这是本视频的灵魂,收工时必须能独立复述:

1. bigram 的 NN 版没有隐层:每个输入字符独立地选出 W 的一行做 softmax。因此**模型能表达的分布族 = "每个当前字符一个任意 categorical 分布" = 计数模型的分布族**。两者参数化不同、容量完全相同。
2. 对这个分布族,NLL 的全局最优解就是**经验条件频率**——正是计数模型直接写下的答案。(计数 = 该优化问题的解析解;梯度下降 = 同一问题的迭代解。)
3. 所以梯度下降收敛后 `W ≈ log(N 的行归一化前计数)`(差一个每行的常数平移,softmax 对平移不变),loss 逼近 2.454。
4. "几乎"而不是"完全":有限迭代、lr、以及正则项让 W 不会精确到达最优。

**这个等价性的价值**:计数法不可扩展——上下文加长一个字符,表就要大 27 倍(#2 开头的动机);而"可微参数 + 梯度下降"这套框架对任意架构通用。视频用一个可以手工验证答案的问题,验证了梯度下降这台机器本身是对的。以后你训练任何新架构,心里都应该有这样一个"已知正确答案的小问题"做 sanity check。

### 2.6 平滑 ≡ 正则化(课前问题 3)

| 计数版 | NN 版 |
|---|---|
| `N + k`(加伪计数) | `+ alpha * (W**2).mean()`(L2 正则) |
| k 越大 → 分布越接近均匀 | W 被拉向 0 → logits 全 0 → softmax 输出**均匀分布** |

两者都是"往均匀分布方向拉,防止把训练集里的偶然当成必然"。同一个统计直觉(别过拟合稀疏计数)在两种参数化下的化身。注意对应关系的方向:**平滑强度 k ↔ 正则强度 alpha**,都是拿"训练集拟合度"换"泛化/稳健"。

### 2.7 从 NN 采样

```python
ix = 0
while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W
    p = logits.exp() / logits.exp().sum(1, keepdim=True)
    ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
    if ix == 0: break
```

与计数版采样唯一的差别:`P[ix]` 换成"现场用 W 算出第 ix 行的分布"。生成结果(同 seed)几乎一样——等价性的又一次可视化确认。

### ⏹ 会话 2 收工自测

- [ ] NN 版训练收敛,loss ≈ 2.48,与计数版几乎相同;能不看笔记讲出 2.5 的四步论证。
- [ ] 口头回答:one-hot @ W 等价于什么?logits 为什么叫 logits?平滑对应 W 的什么约束?
- [ ] 能写出 softmax 两行、NLL 花式索引一行。

---

## Chapter 2(上)复习卡

1. **语言建模的不变量**:词表 + 条件分布 + 采样循环 + 平均 NLL。从 bigram 到 GPT 只换"分布怎么算"。
2. **两个必背数字**:瞎猜 NLL = log 27 ≈ 3.30;bigram 最优 ≈ 2.45。训练任何字符级模型,第 0 步 loss 应≈瞎猜值(初始化检查),收敛值应压过 bigram(容量检查)。
3. **一条恒等链**:one-hot @ W = 选行 = 查表 = embedding。#2 的 `C[X]` 就是它。
4. **一对孪生**:平滑(+k) ≡ L2 正则(αW²)——都朝均匀分布拉。
5. **一个习惯**:所有 `sum/mean` 后面跟广播的地方,检查 keepdim 与归一化轴。
6. 等价性论证(2.5)是面试级问题:"为什么单层 softmax 网络等价于计数 n-gram?" 要能白板作答。

**易错点清单**:
- `P.sum(1)` 忘 keepdim → 静默归一化错轴
- `F.one_hot(...)` 忘 `.float()` → 整型进矩阵乘报错
- `W.grad = None`(或清零)忘写 → 梯度跨步累积(micrograd 老朋友)
- 平均 NLL 忘了除以样本数 → 无法跨数据集比较
- 采样时忘了从 `ix=0`('.')出发、或忘了抽到 0 停止

---

## 与后续章节的钩子

- **→ makemore #2(Chapter 2 下半)**:bigram 的根本瓶颈 = 上下文只有 1 字符且表随上下文长度指数爆炸;解法 = embedding(one-hot@W 的推广)+ 隐层 + 拼接上下文。本讲义 2.2 的恒等式是入场券。
- **→ makemore #3**:"第 0 步 loss ≈ log 27"将成为诊断初始化的第一张图。
- **→ GPT**:采样循环(1.4)与 NLL(1.5)原封不动地出现在莎士比亚生成里;届时回来重读这两节。
