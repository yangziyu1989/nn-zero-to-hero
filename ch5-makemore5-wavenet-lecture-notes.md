# Chapter 5 讲义 · makemore #5 — WaveNet 式层级结构

> 视频:Zero to Hero #6 "Building makemore Part 5: Building a WaveNet"(~1h)
> 对应 study guide 的 1 个会话(可在"容器化改写"处拆两次)。蓝本:DeepMind WaveNet(2016)的层级思想。
> 一句话总结:**上下文加长后,不要让一个隐层"一口吞下"全部字符;像滤波器组一样两两渐进融合,感受野逐层翻倍。** 这是你信号处理主场与深度学习的第一次正面会师。

---

## 全局图景:从"平铺"到"层级"

#2 的 MLP 结构:把 block_size 个字符的 embedding 拼成一个长向量,**一层**线性变换直接压缩。上下文 3 → 8 后(会话课前问题 1),两个问题浮现:

1. **参数问题(轻)**:第一层 fan-in 从 3×n_embd 涨到 8×n_embd,参数线性涨——还扛得住;
2. **结构问题(重)**:8 个字符的全部信息在**单步**里被挤压进一个隐向量,太快太粗暴。字符间的局部结构(双字母组合、音节)没有机会先在低层组织起来,一切都指望一个矩阵一步到位。

WaveNet 的处方:**双字符 → 双双字符 → 八字符**,逐层两两融合。每层只做"把相邻两个片段的表示融合成一个"这件小事,信息被**渐进地**压缩,深层自然获得越来越大的感受野。

顺带,本集也是一次"PyTorch 化"改造练习:把 #3 的手搓层封装成 `torch.nn` 风格的容器(Sequential / Embedding / FlattenConsecutive),学会用 shape 驱动开发。

---

## 会话 · 层级 MLP:结构、形状与 BN 的一个暗雷

### 1.1 基线迁移:block_size 3 → 8

沿用 #3 的网络(已含 Kaiming 初始化 + BatchNorm + 诊断图),只把上下文改成 8:

```
#3 基线 (block=3):  dev ≈ 2.10
平铺 MLP (block=8): dev ≈ 2.02     ← 光加长上下文就已经有利可图
```

这一步同时暗示:上下文里确实还有大量没榨干的信息——值得为它升级架构。

### 1.2 模块化:自己搭一遍 torch.nn 的地基

```python
class Embedding:            # 就是那张 C 表, 前向 = self.weight[IX]
class FlattenConsecutive:   # 见 1.3, 本集唯一的新算子
class Sequential:           # 层的列表, 前向逐层调用
# Linear / BatchNorm1d / Tanh 沿用 #3
```

平铺版模型写成容器形式:

```python
model = Sequential([
  Embedding(vocab_size, n_embd),          # (B, 8) → (B, 8, 10)
  FlattenConsecutive(8),                  # → (B, 80)   一口吞
  Linear(80, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
])
```

### 1.3 FlattenConsecutive:view 的正式上岗(课前问题 4)

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)   # 相邻 n 个时间步的通道拼到一起
        if x.shape[1] == 1:
            x = x.squeeze(1)                  # (B,1,C') → (B,C'), 供最后的线性层
        self.out = x
        return x
```

- `view` 能这么用的前提是内存连续布局:(B,T,C) 里同一样本的时间步相邻存放,把 T 维每 n 个一组折进 C 维是零拷贝重解释——#2 讲的"存储 vs 视图"直接兑现;
- `squeeze(1)` 只在 T 收缩到 1 时用,消掉多余维度。**view 负责重组,squeeze 负责收尾**,这就是两者在 (B,T,C) 重组里的配合。

### 1.4 层级模型:形状全链路(默写级)

```python
model = Sequential([
  Embedding(vocab_size, 10),
  FlattenConsecutive(2), Linear(20, 68), BatchNorm1d(68), Tanh(),
  FlattenConsecutive(2), Linear(136, 68), BatchNorm1d(68), Tanh(),
  FlattenConsecutive(2), Linear(136, 68), BatchNorm1d(68), Tanh(),
  Linear(68, 27),
])
```

```
(B, 8)                     字符索引
 → (B, 8, 10)              embedding
 → (B, 4, 20) → (B, 4, 68) 第一层: 相邻 2 字符融合 (bigram 级特征)
 → (B, 2, 136) → (B, 2, 68) 第二层: 相邻 2 个 bigram 融合 (4-gram 级)
 → (B, 1, 136) → squeeze → (B, 68)  第三层: 全上下文融合 (8-gram)
 → (B, 27)                 logits
```

**中间维度 T 没消失**:线性层只作用于最后一维,(B,4,20)@W(20,68) 是完全合法的批量矩阵乘——PyTorch 把前面所有维都当 batch。同一组权重**滑过每个位置**,这就是权重共享。

**感受野增长**(课前问题 2):每层翻倍,2 → 4 → 8;L 层看到 2^L 个字符。**参数量随深度线性涨,感受野指数涨**——这就是层级结构对平铺结构的根本优势。而且低层特征("qu"、"th" 这类组合)在**所有位置共享**,学一次到处用;平铺 MLP 里同样的知识要在 8 个位置各学一遍。

### 1.5 ⚠ BatchNorm1d 的暗雷:3D 输入的归一化轴

首训后 Karpathy 例行检查各层形状与统计量,发现 BN 的 running_mean 形状是 **(1, 4, 68)**——不对劲:BN 应该**每个通道**一套统计量,即 (1, 1, 68)。

病因:手写的 BatchNorm1d 只对 `dim=0` 求均值;输入是 3D (B, T, C) 时,(B,·) 之外 T 维没有参与归约,变成了"每个 (位置,通道) 对各自归一化"。位置维的样本没被合并,统计量噪声更大,而且语义上错了——同一通道在不同位置应该是同一个特征。

```python
dim = 0 if x.ndim == 2 else (0, 1)      # 修复: 2D 归约 batch, 3D 归约 batch+时间
xmean = x.mean(dim, keepdim=True)
xvar  = x.var(dim, keepdim=True)
```

修复后 dev loss 还小赚一点。**教训两条**:① 又一个"形状合法、不报错、静默错误"的广播类 bug(#2 的 keepdim、#4 的补偿求和,同一家族);② **检查中间张量的形状与统计量**是抓这类 bug 的唯一手段——这正是 #3 仪表盘习惯的延伸。PyTorch 官方 `nn.BatchNorm1d` 对 3D 输入约定是 (N, C, L),通道在中间——用库之前读文档核对轴约定。

### 1.6 战报与开发方法论

```
平铺 (block=8, 22k 参数):     dev ≈ 2.02
层级 (block=8, 76k 参数):     dev ≈ 1.99
BN bug 修复 + 微调:           dev ≈ 1.98~1.99
(继续加大 n_embd/n_hidden 还能压到 ~1.97 以下,代价是训练时间)
```

比数字更重要的是本集示范的**开发节奏**:先在 notebook 里用一个小 batch 走形状(shape-driven prototyping)→ 每步 print shape → 搭好后看各层统计 → 再上真训练。以及一个朴素的超参现实:视频没有系统调参,只演示"怀疑瓶颈 → 改一处 → 对照 dev loss"的循环。

### 1.7 与真 WaveNet 的差距 = 与卷积的关系(课前问题 3,SP 主场)

本集实现和 DeepMind WaveNet 的对应与差距:

- **相同的拓扑**:两两融合的二叉树 = **膨胀因果卷积**(dilated causal convolution)的计算图,dilation 1, 2, 4, …;"因果"= 只看过去(生成模型的硬约束);
- **差距 1(效率)**:我们对每个训练样本独立算一遍整棵树;真正的卷积实现让树在时间轴上**滑动**,相邻位置的中间结果**大量复用**——"for 循环从 Python 挪进 kernel"。这就是卷积的全部本质:**带权重共享的滑动线性层 + 高效实现**;
- **差距 2(结构)**:WaveNet 还有 gated activation(tanh ⊙ sigmoid)与残差/skip 汇聚,本集未做。

**与滤波器组/金字塔的对照**(写进检查清单要求的半页笔记):

| 概念 | 多相/滤波器组 | 本集层级网络 |
|---|---|---|
| 每级操作 | 定常 FIR 滤波 + 抽取(↓2) | 学习到的线性层 + FlattenConsecutive(2) |
| 级联效果 | 频带逐级细分 / 尺度分解 | 感受野逐级翻倍 |
| 系数来源 | 设计(QMF、小波基) | 梯度下降学出来 |
| 级间 | 线性(完美重构可证) | **非线性**(tanh/BN 插在级间) |
| 目标 | 重构/分析 | 判别/生成(压成预测分布) |

一句话:**FlattenConsecutive(2)+Linear 就是一个"学习版的两抽头分析滤波器 + 二抽取",整个网络是一个非线性的、系数可学的金字塔分解**。你设计滤波器组的直觉(混叠、级联增益、感受野=滤波器长度)在这里全部可迁移;不可迁移的是线性系统的可解性——非线性级联只能靠 #3 的仪表盘去"看"。

### ⏹ 收工自测

- [ ] 层级模型训练通过,dev loss 优于平铺版;BN 3D bug 亲手复现并修复。
- [ ] 画出感受野随层数增长示意图(checklist 项)。
- [ ] 半页"卷积 vs 滤波器组"笔记完成。
- [ ] 口头回答:平铺的结构问题?感受野公式?view/squeeze 分工?BN 3D 输入正确的归约轴?

---

## Chapter 5 复习卡

1. **层级公理**:参数线性涨、感受野指数涨、低层特征跨位置共享——三条合起来就是"深度优于宽度"在序列模型上的形态。
2. **卷积 = 权重共享的滑动线性层 + 中间结果复用**。懂了这句,CNN 对你没有新概念,只有新 API。
3. 广播家族 bug 第三案:BN 的 3D 归约轴。破案手段永远是**看形状、看统计量**。
4. (B,T,C) 三件套:线性层只吃最后一维;view 折叠 T→C;squeeze 收掉 T=1。
5. 你的滤波器组直觉 ↔ 层级网络的映射表(1.7)——面试被问"CNN 和信号处理什么关系"时的标准答案。

**易错点清单**:
- FlattenConsecutive 忘 squeeze,最后线性层收到 (B,1,C) 形状错
- BatchNorm1d 3D 输入只归约 dim 0(本集官方 bug,自己写时照样会犯)
- 误以为中间层的线性只能吃 2D 输入而提前 flatten(丢掉层级结构,退化回平铺)
- 感受野算错:是 2^层数,不是 2×层数

---

## 与后续章节的钩子

- **→ GPT 视频**:层级卷积的感受野靠深度**慢慢长**;attention 一步到位——任意两个位置直接通信,感受野=全序列。对比着理解"为什么 Transformer 取代了 WaveNet 式结构做序列建模",以及代价(O(T²))。
- **→ 音频 Phase**:WaveNet/因果卷积在音频生成史上的位置(采样点级自回归),EnCodec/SoundStream 的卷积编码器同宗;你的 mel/滤波器组本行将在项目二正面接上。
