# Chapter 6‒7 讲义 · Let's build GPT — 从矩阵平均技巧到 Transformer

> 视频:Zero to Hero #7 "Let's build GPT: from scratch, in code, spelled out"(~2h)
> 对应 study guide 的 4 个会话;会话 2 是**全系列最重要的一段**(总纲原话:宁可花两倍时间)。数据:tiny shakespeare(~1.1MB 文本)。
> 一句话总结:**attention 不是被发明出来背诵的,它是从"用下三角矩阵乘法做批量加权平均"这个技巧一步步推出来的。** 走通这条推导链,Transformer 对你就不再是图,而是必然。

---

## 全局图景:makemore 管线原封不动,只换大脑

任务仍是字符级语言建模(词表 65:字母+标点),loss 仍是 cross-entropy,采样仍是"查分布→multinomial→反馈"循环——**#1 建立的全部不变量原样适用**。变化只有一个:条件分布 P(next | context) 由 Transformer 计算,上下文长达 block_size=256 个字符,且——与 WaveNet 的逐层扩大不同——**任意两位置一步直连**。

架构总览(写代码前先在纸上画出来):

```
idx (B,T)
 → token_emb(B,T,C) + pos_emb(T,C)
 → [ Block × N ]           Block = 通信(attention) + 计算(FFN), 各带残差+LayerNorm
 → LayerNorm → Linear(C, vocab)
 → logits (B,T,vocab)
```

---

## 会话 1 · 数据管线 + bigram 基线

### 1.1 tokenizer:两个映射而已(课前问题 1)

```python
chars = sorted(list(set(text)))            # 65 个字符
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]    # str → [int]
decode = lambda l: ''.join(itos[i] for i in l)
```

字符级 tokenizer 的最小配置 = **encode/decode 一对映射**。真实 GPT 用子词级 BPE(GPT-2 词表 50257):序列更短、词表更大——**序列长度与词表大小的 trade-off**。BPE 自实现是 Chapter 15 的任务,先留钩子。

### 1.2 batch 的形状:一份数据,多份监督(课前问题 2)

```python
def get_batch(split):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]   for i in ix])   # (B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (B,T) 右移一位
    return x, y
```

x 与 y 错开一位 → **每个位置都是一个训练样本**:位置 t 的任务是"看 x[:t+1],预测 y[t]"。一个 (B,T) batch 含 **B×T** 个样本(8×8=64,不是 8)。这个"一序列多监督"是 Transformer 训练效率的来源之一;同时模型也顺便学会了在 1~block_size 任意长度的上下文下工作(生成时从单字符冷启动全靠它)。

另设 `estimate_loss()`:eval 模式 + `@torch.no_grad()` + 多 batch 平均,train/val 各估一次——#2 的数据纪律,工程化了。

### 1.3 bigram 基线(课前问题 3)

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)        # (B,T,vocab) 直接查出 logits
        loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss
```

与 makemore #1 的 bigram 数学上同一个模型;实现差异(课前问题答案):① one-hot@W 换成 `nn.Embedding` 查表(#2 恒等式);② 包成 `nn.Module`,带 `generate()`,**接口已按"任意长上下文模型"设计**——bigram 只是插在里面的最小大脑,后面逐步换成 Transformer 而管线不动。这是很值得学的工程手法:先搭好可扩展的骨架,用最笨的模型打通全链路。

`generate()`:取**最后一个时间步**的 logits → softmax → multinomial → 拼回 idx 循环。cross_entropy 只吃 2D (N,C),所以有 `view(B*T, C)` 的变形——形状律的日常。

训练(AdamW,lr 1e-3)→ val ≈ **2.5**,生成一团乱码。基线立好,瞎猜值 log65≈4.17 也确认过。

### ⏹ 会话 1 收工自测

- [ ] bigram 在莎士比亚上训练并生成;能答:B×T 个样本怎么数出来的?这个 bigram 与 #1 的两点实现差异?

---

## 会话 2 ★ · self-attention 的数学核心

> 总纲:全系列最重要一段。目标不是"会写",是**能向别人复现推导链**:平均 → 矩阵乘 → softmax mask → 数据依赖的权重 = attention。

### 2.1 出发点:每个位置聚合它之前位置的信息

最笨的聚合:位置 t 取前缀 x[0..t] 的**均值**(丢时序,但先打通"通信"):

```python
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xbow[b,t] = x[b,:t+1].mean(0)      # O(T²) 的显式循环
```

### 2.2 技巧:下三角矩阵乘法 = 批量前缀加权(课前问题 1)

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)   # 每行归一化: 第 t 行 = 前 t+1 个位置各 1/(t+1)
xbow2 = wei @ x                        # (T,T) @ (B,T,C) → (B,T,C), 与循环版完全相等
```

看清这一步:**wei 的第 t 行是"位置 t 从各位置取多少"的权重向量;矩阵乘一次性完成所有位置的加权求和**。下三角形状 = "不许看未来"的结构化表达。所有 for 循环消失,换成一次(可并行、可 GPU 的)matmul。

### 2.3 换一种写法:softmax + mask(为"可学习"做准备)

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))                          # 亲和度, 暂时全 0
wei = wei.masked_fill(tril == 0, float('-inf'))    # 未来位置 → -inf
wei = F.softmax(wei, dim=-1)                       # 每行变概率 → 与 2.2 的均值矩阵相同
xbow3 = wei @ x
```

**为什么用 −inf 而不是 0 填充**(课前问题 3):softmax 里 e^{−inf}=0——未来位置拿到**恰好为零**的权重;若填 0,e⁰=1,未来位置会分走实实在在的概率——**信息泄漏**,自回归模型的死罪。−inf 是"softmax 语言里的不可能"。

这一步的深意:把聚合权重写成 `softmax(亲和度矩阵)` 后,**亲和度从常数 0 换成任何数据算出来的量,整个结构照常工作**——门已经打开。

### 2.4 最后一步:让亲和度由数据决定 → self-attention(课前问题 2)

均匀平均把"the" 和关键人名看得一样重。位置 t 想说:"我在找某类信息(**query**)";每个位置 s 想说:"我含有某类信息(**key**)"。**匹配度 = q·k 点积**——两个向量方向越一致,亲和度越高,这正是你熟悉的相关性/匹配滤波:内积作为相似度。第三个投影 **value**:"如果你要聚合我,请拿走这个"(原始 x 是私有原料,v 是对外发布的通报)。

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)                     # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * head_size**-0.5       # (B,T,T) 亲和度
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v                                        # (B,T,hs)
```

与 2.3 唯一的区别:`wei` 从常数变成 `q @ kᵀ`。**推导链完整版:前缀平均 → 下三角 matmul → softmax(mask) → 把亲和度换成 q·k → attention。** 每一步都只改一个部件——这就是要能向 Claude 复述的那条链。

### 2.5 为什么除以 √d_k(课前问题 4,对照 P1 论文)

q、k 各维独立单位方差时,点积是 head_size 项之和 → **方差 ≈ head_size**。方差大的 logits 进 softmax 会怎样——#3 的饱和直觉直接迁移:softmax 输出趋向 one-hot,每个位置只盯一个位置看,且梯度极小(softmax 在极端区平坦)。除以 √d_k 把方差拉回 1,softmax 保持"温和分散",训练初期尤其关键。**这是 #3"控制预激活方差"的思想在 attention 里的精确复刻**——初始化控制的是层的方差,scaling 控制的是亲和度的方差。

### 2.6 Karpathy 的注解清单(每条都值得复述)

- **attention = 有向图上的通信机制**:节点带信息,按亲和度加权收取入边消息;下三角 = "第 t 个节点只连前 t 个";
- **attention 没有空间概念**:对位置的排列天然不变,所以必须**注入位置信息** → `pos_emb`(本实现:学习的位置嵌入表,(T,C),加到 token embedding 上);对比:卷积天生带空间结构,attention 要手工补;
- batch 维完全独立:样本间从不通信;
- **decoder vs encoder**:有 tril mask = decoder(自回归生成);去掉 mask = encoder(如 BERT、翻译的源端),代码只差一行;
- **self vs cross attention**:q、k、v 同源 = self;q 来自本序列、k/v 来自别处(如编码器输出)= cross——架构不变,只换数据来源。

### ⏹ 会话 2 收工自测

- [ ] 单头 attention 从零写出并跑通。
- [ ] 不看资料,完整讲一遍 2.1→2.4 的推导链(向 Claude 讲,卡壳即重看)。
- [ ] 口头回答:−inf vs 0?√d_k 不除会怎样(两个后果:one-hot 化+梯度小)?decoder/encoder/cross 三个变体各改哪里?

---

## 会话 3 · 多头 + FFN + Block 堆叠

### 3.1 多头:并行的多路通信(课前问题 1)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # 拼回 n_embd
        return self.proj(out)
```

- 多出的表达能力来自哪:单头每个位置只能形成**一种**注意力分布(一套"我在找什么");多头 = **多个独立子空间各自通信**——一个头管句法邻接、一个头管远处人名、一个头管标点节奏(比喻)。头之间**独立**:各自的 q/k/v 投影与注意力矩阵;**共享**:输入 x 和后面的拼接投影;
- n_embd=384、6 头 → head_size=64:总计算量与单个 384 维大头同级,却得到 6 个独立的注意力模式——**免费的多样性**;
- **拼接后为什么还要 proj**(checklist 问题):各头的输出只是并排放着,互不相识;proj 做一次线性混合,把 6 路通报综合成对残差流的**一份**统一更新,也校准回残差流的"坐标系"。

### 3.2 FFN:通信之后,各自思考(课前问题 2)

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout),
        )
```

**分工**:attention 是 token 之间**唯一**交换信息的地方(通信);FFN 逐 token 独立运算(计算)——"收完消息后,每个节点自己消化"。没有 FFN 的纯 attention 网络效果差:数据在线性聚合里打转,缺少逐位置的非线性加工。4× 扩宽是原论文惯例(先升维思考再降维回流)。一个 Block = 通信 + 计算,一轮完整的"开会 + 干活"。

### 3.3 残差连接:梯度高速公路(课前问题 3)

网络堆到几个 Block 后直接训——变差了:深网络优化难(#3/#4 亲历过的梯度问题)。第一味药:

```python
x = x + self.sa(self.ln1(x))       # 不是 x = self.sa(...)
x = x + self.ffwd(self.ln2(x))
```

从梯度流看(#4 手推过 `+` 的反向):**加法节点把上游梯度原样分发给两条支路**——恒等支路上梯度从 loss 直达底层,无衰减、无变形;各子层支路是"旁路修正"。初始时子层输出小,网络近似恒等映射(浅网络般好训),训练中支路逐渐"上线"。残差把"学习 H(x)"改成"学习修正量 H(x)−x",也把"深"从优化障碍变成累加的精修步数。

### 3.4 LayerNorm:归一化轴的换位(课前问题 4)

第二味药。与你手写过的 BatchNorm 对照:

| | BatchNorm(#3) | LayerNorm(本集) |
|---|---|---|
| 归一化轴 | batch 维(每特征跨样本) | 特征维(**每 token 内部**跨通道) |
| 样本间耦合 | 有(整套 running stats 因它而生) | **无** |
| 训练/推理差异 | 有(running stats) | 无,同一公式 |
| 变长序列/小 batch | 别扭 | 天然适配 |

序列模型偏爱 LayerNorm 的原因即上表右列:**逐 token 自给自足**,不需要 batch 统计量,推理单样本、变长、自回归逐步生成都毫无别扭。γ/β 仍在(#3 的理由原样成立)。

**Pre-norm 惯例**:本实现把 LN 放在子层**之前**(`x + sa(ln(x))`),与原论文的 post-norm(`ln(x + sa(x))`)不同——pre-norm 让恒等支路完全干净,深堆更稳,是现代 GPT 的标准做法。

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_head, n_embd//n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

### ⏹ 会话 3 收工自测

- [ ] 完整 Block 堆叠(3~4 层小配置)训练通过,val 明显优于单头版。
- [ ] 口头回答:多头独立/共享什么?拼接后 proj 干什么?通信/计算分工?残差的梯度论证(用 `+` 分发规则说)?LN vs BN 的轴与三条工程后果?

---

## 会话 4 · Dropout、放大与收尾

### 4.1 dropout:训练时随机断线(课前问题 1)

三个安放位置(Karpathy 的选择,也是通行做法):

1. attention 权重 wei 经 softmax 之后(随机切断一部分通信边);
2. 多头拼接投影 proj 之后(每个残差支路汇入前);
3. FFN 输出之后(同上)。

**训练时**:按概率 p 随机置零并按 1/(1−p) 放大存活项(inverted dropout,保持期望)——等效于每步训练一个随机子网络,最终模型 ≈ 子网络集成,抑制过拟合。**推理时**:恒等,全网络上岗。`model.train()/eval()` 切换的另一半含义(BN 之外)就是它。放大模型时才开(小模型欠拟合,不需要)。

### 4.2 放大:哪些超参必须联动(课前问题 2)

最终配置:

```
n_embd 384, n_head 6, n_layer 6, block_size 256,
batch_size 64, dropout 0.2, lr 3e-4 (调低!)
→ ~10M 参数, val loss ≈ 1.48 (基线 bigram 2.5)
```

联动规则:

- **模型变大 → lr 调低**(update/param 直觉:参数尺度与梯度结构都变了;大模型用 3e-4 这类"保命值");
- **容量变大 → dropout 打开**(过拟合风险随容量涨);
- **block_size 变大 → 位置嵌入表、tril buffer、显存、每步耗时全部跟着变**;batch_size 与显存对冲;
- n_embd 必须能被 n_head 整除(head_size = n_embd/n_head)。

生成效果:能排出剧本的**版式**(角色名: 台词、换行、古英语腔),内容仍是胡话——10M 参数、字符级、单 GPU 几分钟,形式先于内容被学会。**评估直觉:形式易学,语义贵。**

### 4.3 与 nanoGPT/真 GPT 的差距清单(课前问题 3,Phase 2 钩子)

- tokenizer:字符级 → **BPE**(Chapter 15 自实现);
- 训练配方:恒定 lr → **warmup + cosine decay**;无 → **weight decay 分组、grad clip**;权重初始化方案(残差支路缩放);
- 系统:单卡 fp32 → **mixed precision、DDP 多卡、torch.compile、flash attention**;checkpoint/resume(本 Chapter 工程任务先做);
- 结构小项:**weight tying**(输入嵌入与输出投影共享)等;
- 规模:10M/1MB 数据 → 124M/数十 GB 数据(Chapter 14 的 GPT-2 复现清账)。

### 4.4 后记:从 GPT 到 ChatGPT

本集做的是 **pretraining**(下一 token 预测,学会"语言+知识+形式")。到 ChatGPT 还差对齐三部曲:**SFT**(问答格式微调)→ **奖励模型**(人类偏好打分)→ **RLHF/PPO**。文档补全器 ≠ 助手,差的是这三步——Phase 2 之后的世界,先存档。

### ⏹ 会话 4 收工自测

- [ ] 放大版训练完成(远程,含 checkpoint/resume),生成质量肉眼提升。
- [ ] 口头回答:dropout 三个位置与训练/推理差异?放大时的三组联动?与 nanoGPT 差距任说五条。

---

## Chapter 6‒7 复习卡(闭卷重构 GPT 前过一遍)

1. **推导链(必须能脱稿讲)**:前缀平均 → tril matmul 批量化 → softmax+mask 改写 → 亲和度换成 q·kᵀ/√d_k → 单头 attention → 多头+proj → +FFN(通信/计算)→ +残差+preLN(可堆深)→ +dropout(可放大)。每个箭头只改一个部件。
2. **三个"为什么"高频面试题**:−inf mask(softmax 语言的零)、√d_k(方差→softmax 饱和→梯度)、LN vs BN(轴+无 batch 耦合)。答案全部接在 #3/#4 的地基上。
3. **形状主线**:(B,T) → +emb → (B,T,C) → Block×N(内部 (B,T,T) 注意力矩阵)→ (B,T,vocab) → view(B·T, vocab) 进 CE。
4. **B×T 个监督信号**、生成时 crop 到 block_size、`register_buffer`(tril 不是参数)——三个易忘的工程细节。
5. ★ 闭卷重构清单(Chapter 7):空白文件写出 Head → MultiHead → FFN → Block → GPT + 训练循环 + generate,小配置在莎士比亚上 val < 2.0。

**易错点清单**:
- mask 忘了 `[:T,:T]` 切片(生成时 T < block_size 就崩)
- `wei @ v` 写成 `wei @ x`(聚合了原料而非通报,能跑但错)
- softmax 的 dim 忘写或写错(必须是最后一维)
- 残差写成 `x = self.sa(ln(x))` 丢了 `x +`(深网瞬间训不动,症状回看 #3)
- generate 里取 logits 忘了 `[:, -1, :]`
- lr 沿用小模型的 1e-3 放大后直接发散

---

## 与后续章节的钩子

- **→ Chapter 7**:官方 exercises + 闭卷重构 + 向 Claude 无提词讲解 attention 全流程(Phase 0 毕业礼)。
- **→ Chapter 14‒16(GPT-2 复现)**:4.3 的差距清单就是任务书,逐条清账。
- **→ Phase 2 audio LM**:"换模态不换架构"——同一个 GPT,token 从字符换成 EnCodec 音频码本,你的简历核心项目在此发芽。
