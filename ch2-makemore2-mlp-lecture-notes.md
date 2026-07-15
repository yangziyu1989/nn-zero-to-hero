# Chapter 2 讲义(下) · makemore #2 — MLP 语言模型

> 视频:Zero to Hero #3 "Building makemore Part 2: MLP"(~1.3h)
> 对应 study guide 的 2 个会话。蓝本:Bengio et al. 2003《A Neural Probabilistic Language Model》。
> 一句话总结:**用 embedding + 隐层突破 bigram 的"一字符上下文"天花板;同时装上真正训练所需的三件工具——minibatch、学习率搜索、train/dev/test 划分。** 从这个视频起,你写的才是"正常的"深度学习代码。

---

## 全局图景:为什么必须放弃计数

bigram 的根本瓶颈(会话 1 课前问题 1):**上下文只有 1 个字符**。想看更远?计数表的行数是 27^(上下文长度):

| 上下文 | 表的行数 |
|---|---|
| 1 字符 | 27 |
| 2 字符 | 729 |
| 3 字符 | 19,683 |
| 8 字符 | ~2800 亿 |

指数爆炸带来两个死刑:存不下,以及**绝大多数上下文在训练集里一次都没出现过**——没有计数可查,模型对没见过的上下文彻底哑火。

Bengio 2003 的解法(本视频的全部内容):把每个字符嵌入到低维连续空间(embedding),让**相似的上下文共享统计强度**。训练集里见过 "the cat was running",就自动对 "a dog was running" 有了把握——因为 a/the、cat/dog 的 embedding 相近。参数量随上下文长度**线性**增长,泛化靠**空间中的邻近**而非查表命中。

---

## 会话 1 · 上下文窗口 + embedding + 隐层

### 1.1 构建数据集:滑动窗口

```python
block_size = 3                       # 上下文长度: 用 3 个字符预测第 4 个
X, Y = [], []
for w in words:
    context = [0] * block_size       # 用 '.' (0 号) 填充起始
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix] # 窗口左移
X = torch.tensor(X)                  # (N, 3)  N ≈ 228k
Y = torch.tensor(Y)                  # (N,)
```

`'...' → e`、`'..e' → m`、`'.em' → m`……每个名字贡献 len+1 个样本。起始填充 `.` 优雅地统一了"上下文不足"的边界情况。

### 1.2 embedding:one-hot @ C 的高效写法(课前问题 2)

```python
C = torch.randn((27, 2))     # 27 个字符, 每个嵌入到 2 维
emb = C[X]                   # (N, 3, 2) —— 花式索引直接查表
```

上一份讲义焊牢的恒等式在此兑现:**`one_hot(ix) @ C ≡ C[ix]`**。one-hot 乘矩阵就是选行,那不如直接按下标取行——省掉一次全是 0 的乘法。所以:

- embedding 层不是新概念,它就是 bigram NN 里的第一层线性层,只是实现从"矩阵乘"优化成了"查表";
- 反过来说,**embedding 表 C 是可训练参数**,梯度会流回它(被索引到的行才有梯度——#4 会手推这一点)。

PyTorch 花式索引强大之处:`C[X]` 里 X 是 (N,3) 整数张量,结果直接是 (N,3,2)——每个位置的下标替换成对应的 2 维向量。

### 1.3 拼接与 view:形状怎么变(课前问题 3)

隐层要的输入是每个样本一个平的向量:3 个字符 × 2 维 = 6 维。

```python
# 笨办法: torch.cat(torch.unbind(emb, 1), 1)  → 产生新内存
# 正确姿势:
emb.view(-1, 6)              # (N, 3, 2) → (N, 6), 零拷贝
```

**view 为什么免费**:张量底层是一维连续存储 + (shape, stride) 解释器;view 只改解释器不动数据。`cat` 则真的分配新内存拼数据。`-1` 让 PyTorch 自己推断该维大小。这套"存储 vs 视图"的心智模型在 #5(WaveNet 的 FlattenConsecutive)会再次吃重。

形状变换全链路(默写级):

```
X (N,3) --C[X]--> (N,3,2) --view--> (N,6) --@W1(6,100)+b1--> (N,100) --tanh-->
(N,100) --@W2(100,27)+b2--> logits (N,27)
```

### 1.4 隐层与前向

```python
W1 = torch.randn((6, 100));  b1 = torch.randn(100)
W2 = torch.randn((100, 27)); b2 = torch.randn(27)

h = torch.tanh(emb.view(-1, 6) @ W1 + b1)   # (N, 100)
logits = h @ W2 + b2                        # (N, 27)
loss = F.cross_entropy(logits, Y)
```

注意 `+ b1` 又是一次广播:(N,100) + (100,) → 右对齐,b1 扩成 (1,100) 逐行加——这次是**正确**的广播,对照上份讲义的 keepdim 陷阱想一遍为什么。

### 1.5 F.cross_entropy:为什么不再手写 softmax + NLL

三个理由(以后写任何分类模型都适用):

1. **数值稳定**:手写版 `logits.exp()` 在 logit=100 时上溢成 inf → nan。cross_entropy 内部先减去每行最大值(softmax 对平移不变,#4 会手推这个不变性),彻底免疫上溢。
2. **效率**:fused kernel,不物化 counts/probs 等中间张量,前向反向都快。
3. **反向简洁**:softmax+CE 合并后的梯度干净得惊人(p − y,#4 的主角)。

规则:**教学时手写一次,生产中永远用库函数。**

### ⏹ 会话 1 收工自测

- [ ] 前向跑通,loss 能算出来;能默写 1.3 的形状链。
- [ ] 口头回答:计数法为什么在长上下文下立刻死亡?C[X] 与 one-hot 的关系?view 为什么零拷贝?

---

## 会话 2 · minibatch + 学习率搜索 + 数据划分

### 2.1 minibatch:嘈杂但便宜的梯度(课前问题 1)

228k 样本全量算一次梯度太贵。改成每步随机抽 32 个:

```python
ix = torch.randint(0, X.shape[0], (32,))
emb = C[X[ix]]                     # 只对 32 个样本前向/反向
...
loss = F.cross_entropy(logits, Y[ix])
```

梯度"错"了吗?——它是全量梯度的**无偏估计**:期望方向正确,只是带噪声。而优化根本不需要每步都精确,只需要**大体朝下**。Karpathy 的原话值得背:

> "It is much better to have an approximate gradient and take many steps, than an exact gradient and take few steps."

对你的 SP 直觉:这就是 LMS 自适应滤波 vs 批量最小二乘——随机梯度本身就是你用了多年的东西。噪声甚至有正面作用(帮助跳出尖锐极小值),#3 的 BatchNorm 还会主动利用 batch 噪声。

### 2.2 学习率搜索:一次扫描代替瞎试(课前问题 2)

系统方法,三步:

```python
lre = torch.linspace(-3, 0, 1000)   # 指数从 -3 到 0
lrs = 10**lre                        # lr 从 0.001 到 1, 对数均匀
# 训练 1000 步, 第 i 步用 lrs[i], 记录 (lre[i], loss)
plt.plot(lrei, lossi)                # 横轴是指数!
```

- 先用两端试探定出"太小(loss 几乎不动)"和"太大(loss 爆炸)"的边界,再在其间**按指数均匀**铺 lr;
- loss-vs-指数曲线呈一个谷:左边平缓下降(lr 太小),谷底一段平坦(好区间),右边翘起(不稳定)。取谷底,这里 ≈ **0.1**;
- 训练后期手动 decay(0.1 → 0.01)再榨最后一点 loss。

为什么对数轴:lr 的自然尺度是数量级,0.01→0.02 与 0.1→0.2 才是"等距"的改动。

### 2.3 train/dev/test:三个集合回答三个问题(课前问题 3)

```python
random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
# 80% train / 10% dev(val) / 10% test —— 注意按"词"切, 不是按 bigram 切
```

| 集合 | 回答的问题 | 使用频率 |
|---|---|---|
| train | 参数(W, b, C)往哪调 | 每一步 |
| dev / val | 超参数(隐层大小、embedding 维度、lr、正则)哪组好 | 每次调超参 |
| test | 最终模型的真实泛化水平 | **极少数几次** |

只有 train/test 两份会发生什么:你会拿 test 调超参,test 事实上变成了 dev——**从此再没有任何一份数据能诚实回答"上线后表现如何"**。test 每被看一次就泄漏一分,这是实验纪律,不是形式主义。

诊断规则(以后所有项目通用):

- `train ≈ dev`:**欠拟合**,模型容量是瓶颈 → 加大网络;
- `train ≪ dev`:**过拟合** → 更多数据/正则/早停/缩小网络。

本视频初期 train ≈ dev ≈ 2.3 → 判定欠拟合 → 放大网络。

### 2.4 找瓶颈 → 放大:embedding 可视化

隐层 100→300 提升有限,怀疑瓶颈在 **2 维 embedding 太挤**。放大前先看一眼(2 维恰好可以直接散点):

```python
plt.scatter(C[:,0].data, C[:,1].data)
# 每个点标上 itos[i]
```

训练后的 C 里,**元音 a/e/i/o/u 聚成一团**,q、'.' 远离大部队——网络自己发现了"元音"这个概念,没人告诉过它。这是"embedding 空间共享统计强度"最直观的一次目击。

放大配置:embedding 2→**10** 维,隐层 →**200**,拼接后输入 30 维:

```
最终 dev loss ≈ 2.17   ( bigram 基线 2.45, 瞎猜 3.30 )
```

采样出的名字开始"像词"了:更长的上下文 + 连续空间泛化,肉眼可感。

### 2.5 〔工程〕远程首航

按 checklist,本次训练在 AutoDL 上完成:SSH → tmux 里跑训练 → 断开重连训练仍在 → checkpoint scp 回本地 → 流程写入 `docs/remote-workflow.md`。模型本身很小,拿它练的是**流程脱敏**——此后所有训练默认远程。

### ⏹ 会话 2 收工自测

- [ ] 完整训练跑通,dev loss < 2.45(压过 bigram 基线)。
- [ ] 口头回答:minibatch 梯度为什么"错了也能用"?lr 扫描三步法?只有 train/test 会出什么事?
- [ ] train≈dev 和 train≪dev 分别诊断为什么、开什么药方?

---

## Chapter 2(下)复习卡

1. **一条主线**:计数表指数爆炸 → embedding 连续空间线性增长 + 邻近泛化。这是从 n-gram 到神经语言模型的历史转折(Bengio 2003),你亲手走了一遍。
2. **形状链默写**:(N,3) → C[X] → (N,3,2) → view(N,6) → tanh(·@W1+b1) → (N,100) → @W2+b2 → (N,27)。
3. **三件工具从此常驻**:minibatch(无偏噪声梯度)、对数轴 lr 扫描、train/dev/test 纪律。
4. **两个诊断反射**:train≈dev=欠拟合,train≪dev=过拟合;lr 曲线谷底取值。
5. cross_entropy 三理由:稳定(减 max)、效率(fused)、反向简洁(p−y)。

**易错点清单**:
- 数据划分按 bigram/样本切而不是按**词**切(同一个词的样本泄漏到两侧)
- `C[X]` 后忘了 view 就直接 @W1(形状错)或用 cat(慢)
- lr 扫描画图时横轴用了 lr 本身而不是指数(谷被压扁看不出来)
- 训练循环里 forward 用了全量 X 而反向只想用 minibatch(loss 与梯度不对应)
- 评估 dev loss 时忘了用**全量 dev**、或忘了 no_grad(慢+吃显存)

---

## 与后续章节的钩子

- **→ #3**:本视频初始化仍是随手 `randn`——第 0 步 loss 远高于瞎猜值 3.30,还有 hockey-stick 曲线和 tanh 饱和,全部按下未表。#3 开场就清算这笔账。
- **→ #4**:`C[X]` 的花式索引反向怎么走(同一行被索引多次 → 梯度累加)、广播的反向补偿,都会逐张量手推。
- **→ #5**:block_size 3→8,平铺 MLP 一口吞 8 字符太粗暴 → WaveNet 层级融合;view 的心法直接复用。
