# Chapter 1 讲义 · micrograd — 反向传播的全部秘密

> 视频:Zero to Hero #1 "The spelled-out intro to neural networks and backpropagation: building micrograd"(~2.5h)
> 对应 study guide 的 3 个会话;本讲义按会话组织,每节末附课前问题参考答案与自测。
> 一句话总结:**神经网络训练 = 在计算图上递归应用链式法则,再沿负梯度方向走一小步。** 整个视频只讲这一件事。

---

## 全局图景:micrograd 是什么、不是什么

- micrograd 是一个 **标量级自动求导引擎(autograd engine)**:每个节点是一个单独的浮点数,不是张量。PyTorch 做的事在数学上完全一样,只是把标量打包成张量以便并行加速。
- 训练一个神经网络只需要两样东西:**前向计算 loss** + **反向计算梯度**。micrograd 全部代码 ≈ 100 行,说明反向传播本身很简单——复杂的只是围绕它的工程。
- 心智模型:任何表达式都是一张 **有向无环图(DAG)**,叶子是输入/参数,根是输出(最终是 loss)。前向沿边算值,反向沿边传梯度。

---

## 会话 1 · 导数直觉 → Value 类与手动反向

### 1.1 导数 = 敏感度,只用一个工具:数值微分

从一个标量函数开始:

```python
def f(x):
    return 3*x**2 - 4*x + 5

h = 0.0001
x = 3.0
(f(x + h) - f(x)) / h   # ≈ 14 = 6x - 4 在 x=3 处的值
```

- 导数的操作性定义:**把输入轻推(nudge)一点,看输出以多大比例响应**。正负号 = 响应方向,大小 = 敏感度。
- 对信号处理背景的你:这就是系统对参数的**灵敏度分析**。整个深度学习中"梯度"一词都可以读作"loss 对该参数的敏感度"。
- 视频刻意不用符号求导:神经网络里没人手写解析导数,一切靠链式法则在图上组合**局部导数**。

多输入的情形——对每个输入分别轻推:

```python
a, b, c = 2.0, -3.0, 10.0
d = a*b + c        # d = 4
# 分别 nudge a / b / c,重算 d:
# ∂d/∂a = b = -3   (a 增大, d 减小, 因为 b 是负的)
# ∂d/∂b = a = 2
# ∂d/∂c = 1
```

课前问题 1 的答案就在这:**多输入表达式的敏感度 = 固定其它输入,逐个 nudge,每次一次前向差分。** 这也是后面所有"gradient check"的原理。

### 1.2 Value 类:把表达式记成图

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0            # ∂L/∂self, 初始为 0 = "不影响输出"
        self._prev = set(_children) # 谁生成了我(图的边)
        self._op = _op              # 什么运算生成了我
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

关键设计,对应课前问题 3:

- **data 与 grad 必须分开存**:data 是前向的值,grad 是反向传回来的 ∂L/∂节点。前向建图算 data,反向按图填 grad,两条通道互不覆盖。
- grad 初始化为 0.0 的语义:"在证明它有影响之前,假设它不影响 loss"。(0 还有第二个作用——支持累加,见会话 2 的 `+=`。)
- `_prev` 记录父子关系,使表达式成为可遍历的 DAG——这是能"自动"反向的前提。graphviz 的 `draw_dot` 只是把这张图画出来。

### 1.3 手动反向传播一个小表达式

视频的标准例子(建议默写):

```python
a = Value(2.0,  label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'   # e = -6
d = e + c; d.label = 'd'   # d = 4
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'   # L = -8
```

从根往叶子填 grad(链式法则 = 上游梯度 × 本地梯度):

| 节点 | 推导 | grad |
|---|---|---|
| L | ∂L/∂L | **1** |
| f | ∂L/∂f = d | **4** |
| d | ∂L/∂d = f | **-2** |
| c | ∂L/∂c = ∂L/∂d · ∂d/∂c = -2 · 1 | **-2** |
| e | ∂L/∂e = -2 · 1 | **-2** |
| a | ∂L/∂a = ∂L/∂e · ∂e/∂a = -2 · b | **6** |
| b | ∂L/∂b = -2 · a | **-4** |

两个必须内化的**局部规则**(以后读任何反向传播代码都靠它们):

- **`+` 节点是"梯度分发器"**:本地导数为 1,把上游梯度原样路由给每个输入。
- **`*` 节点是"交换器"**:给一个输入的梯度 = 上游梯度 × **另一个**输入的前向值。

链式法则在图上的对应(课前问题 2):**沿着从 loss 到该节点的路径,把每条边的局部导数连乘;多条路径则相加。** "局部梯度"指一个运算节点只看自己输入输出的导数(如 `+` 的 1、`*` 的另一操作数),它不需要知道全图——这正是反向传播可以模块化、每个 op 各管各的原因。

验证方式:nudge `a.data += 0.001 * a.grad`,重跑前向,L 应该朝变大的方向动——这也是"梯度指向使输出增大的方向"的第一次演示。

### 1.4 一个神经元的前向

```python
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)          # 刻意取值让 tanh 输出好看
n = x1*w1 + x2*w2 + b                  # 预激活
o = n.tanh()                           # 激活, o ≈ 0.7071
```

tanh 作为原子 op 加入 Value(只需知道本地导数):

```python
def tanh(self):
    t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
    out = Value(t, (self,), 'tanh')
    # 反向时: self.grad += (1 - t**2) * out.grad
    return out
```

手算这条链:`o.grad=1` → `n.grad = 1 - o² = 0.5` → `+` 节点分发 → `*` 节点交换,得到 `w1.grad = 0.5 · x1 = 1.0`,`w2.grad = 0.5 · x2 = 0`。注意 **x2=0 导致 w2 梯度为 0**:输入为 0 时对应权重"学不到东西",这个直觉后面(死神经元、饱和)会反复出现。

### ⏹ 会话 1 收工自测

- [ ] 不看资料,手算 1.3 表格里的全部 grad,并说出 `+`/`*` 两条局部规则。
- [ ] 能解释:为什么 data 和 grad 必须是两个字段?grad 为什么初始化成 0?

---

## 会话 2 · backward() 自动化 → tanh 分解 → 梯度累积 bug

### 2.1 每个 op 自带 _backward 闭包

把会话 1 的手算规则塞进各运算内部:

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad  += 1.0 * out.grad     # + 分发
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad  += other.data * out.grad   # * 交换
        other.grad += self.data  * out.grad
    out._backward = _backward
    return out
```

注意 `_backward` 是**闭包**:它捕获了 self/other/out,反向时不需要任何额外参数。每个 op 只负责"把 out.grad 按本地导数分给输入"——这就是模块化的链式法则。

### 2.2 拓扑排序:为什么必须按序反向(课前问题 1)

```python
def backward(self):
    topo, visited = [], set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)          # 后序:孩子全进来了自己才进
    build_topo(self)

    self.grad = 1.0                 # 种子: ∂L/∂L = 1
    for node in reversed(topo):     # 从根往叶子
        node._backward()
```

- 一个节点的 `_backward()` 做的事是"把**我的 grad**分发给输入"。所以调用它时,**我的 grad 必须已经收齐了所有来自上游的贡献**。
- 拓扑序保证:处理某节点时,所有依赖它的节点(下游/更靠近 loss 的)都已处理完。
- 乱序会错在哪:若某节点在 grad 只收到一半时就执行 `_backward()`,它分发下去的就是**不完整的梯度**,且事后无法补救——下游全错。症状通常是梯度数值"差一部分",而不是明显崩溃,非常隐蔽。

### 2.3 ★ 梯度累积 bug:`+=` vs `=`(课前问题 2)

```python
a = Value(3.0)
b = a + a          # ∂b/∂a 应该是 2
b.backward()
a.grad             # 若 _backward 里写 = :得到 1, 错!
```

- 一个节点被使用多次 = 图里它到 loss 有**多条路径**。多元链式法则说:总梯度 = 各路径贡献**之和**。
- 若 `_backward` 里写 `self.grad = ...`,第二条路径的写入会**覆盖**第一条,梯度偏小;写 `+=` 才是求和。
- 这个 bug 的险恶之处:代码能跑、loss 也在降,只是某些共享变量(后面 = 共享参数,如 embedding 行、RNN 权重)学得系统性偏慢/偏错。
- 副作用:既然反向靠累加,**每轮迭代前必须把 grad 清零**——这就是 PyTorch `zero_grad()` 的由来,会话 3 再撞一次。

### 2.4 tanh:原子实现 vs 分解实现(课前问题 3)

补齐 `exp`、`__pow__`(除法 = `* other**-1`)、`__neg__/__sub__`、`__radd__/__rmul__` 后,可以把 tanh 拆开写:

```python
# 原子版
o = n.tanh()
# 分解版: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
e = (2*n).exp()
o = (e - 1) / (e + 1)
```

两种实现反向出来的 `n.grad` **完全一致**。结论:

- 只要每个 op 的**本地导数**正确,链式法则不关心你在哪个"抽象层级"切分运算——一个 op 可以是 tanh 这么大,也可以是 exp 这么小。
- 这就是深度学习框架的设计自由度:PyTorch 里既有细粒度算子,也有 fused kernel(如 `F.softmax`、fused attention),数学上等价,差别只在效率与数值稳定性。
- 顺带学到 Python 运算符重载的工程细节:`2 * a` 需要 `__rmul__`,`a + 1` 需要把 int 包装成 Value(`other = other if isinstance(other, Value) else Value(other)`)。

### 2.5 与 PyTorch 对照

```python
import torch
x1 = torch.Tensor([2.0]).double();  x1.requires_grad = True
# ... 同样搭 n = x1*w1 + x2*w2 + b; o = torch.tanh(n)
o.backward()
x1.grad.item()      # 与 micrograd 完全一致
```

对应关系(会话 3 课前问题 3 的答案,提前给出):**micrograd 的 Value ≈ PyTorch 的零维/单元素 tensor(requires_grad=True)**。同样有 `.data`、`.grad`、`.backward()`,同样在背后建图、拓扑排序、按 op 的本地导数累加。PyTorch 多出来的只是:张量化(并行)、GPU、庞大的算子库。

### ⏹ 会话 2 收工自测

- [ ] `backward()` 全自动跑通,结果与会话 1 手算一致。
- [ ] 口头回答:拓扑排序防什么错?`+=` 防什么 bug?两者分别对应链式法则的哪个性质?
  (答案:拓扑序保证分发时自身梯度已收齐;`+=` 实现多路径梯度求和,即多元链式法则。)

---

## 会话 3 · Neuron/Layer/MLP + 训练循环 + PyTorch 对照

### 3.1 三层抽象:Neuron → Layer → MLP

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):            # MLP(3, [4, 4, 1])
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

- 每层只是"若干个独立神经元并排",MLP 只是"层的串联"——`x = layer(x)` 这一行就是"深度"的全部含义。
- `parameters()` 层层收集,为的是训练循环里能**一把抓到所有可调参数**(PyTorch `model.parameters()` 同源)。
- `MLP(3, [4,4,1])`:3 维输入、两个隐层各 4 个神经元、1 维输出,共 41 个参数。

### 3.2 数据与 loss

```python
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]     # 二分类风味的回归目标

ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))   # MSE
```

loss 是**把"预测好不好"压缩成一个标量**的装置——只有标量才能求梯度、才能"下降"。MSE 只是最简单的选择,下一章会换成 NLL。

### 3.3 ★ 训练循环四步(课前问题 1)

```python
for k in range(20):
    # 1. forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    # 2. zero grad   ← 最容易忘!
    for p in n.parameters():
        p.grad = 0.0
    # 3. backward
    loss.backward()
    # 4. update (gradient descent)
    for p in n.parameters():
        p.data += -0.05 * p.grad     # 负号:朝 loss 减小的方向
    print(k, loss.data)
```

- **四步:forward → zero_grad → backward → update。** 最容易忘的是 zero_grad——因为 2.3 里我们刚刚把反向传播设计成"累加"。
- 忘掉 zero_grad 的症状:梯度跨迭代不断累积,等效步长越来越大,loss 先降后疯狂震荡/发散;而且它是"看起来在训练"的 bug,Karpathy 本人在视频里都(故意)先犯了一次。
- update 的负号:grad 指向 loss **增大**方向,所以要走反方向。步长 = 学习率。

### 3.4 学习率的形状(课前问题 2)

- **过大**:一步跨过谷底,loss 震荡甚至指数式发散(数值上先看到 loss 上蹿下跳)。
- **过小**:loss 单调但下降极慢,曲线近乎平线。
- 视频做法:手动试 0.01 → 0.1 → 观察,后期手动调小(简易 lr decay)。系统性的学习率搜索留到 makemore #2。

### 3.5 micrograd ↔ PyTorch 对照表(课前问题 3)

| micrograd | PyTorch | 备注 |
|---|---|---|
| `Value(2.0)` | `torch.tensor(2.0, requires_grad=True)` | Value = 标量 tensor |
| `.data` / `.grad` | `.data` / `.grad` | 同名同义 |
| `L.backward()` | `loss.backward()` | 同样是拓扑序 + 局部导数累加 |
| `p.grad = 0.0` 循环 | `optimizer.zero_grad()` | 同一件事 |
| `p.data += -lr * p.grad` | `optimizer.step()`(SGD) | 同一件事 |
| 手写 `_backward` | autograd 内置 op 库 | 数学等价,张量化+GPU |

**最本质的对应:两者都是"建图 → 反向 → 更新"的同一套机制;PyTorch 只是把标量换成张量并工程化。** 学完 micrograd,PyTorch 的 autograd 对你不再是黑盒。

### ⏹ 会话 3 收工自测

- [ ] 二层 MLP 在 4 样本数据集上 loss 稳定下降;PyTorch 对照段跑通。
- [ ] 口头回答训练四步、zero_grad 症状、lr 过大/过小的 loss 形状。

---

## Chapter 1 复习卡(闭卷重构前过一遍)

1. **反向传播 = 拓扑序遍历 + 每个 op 的本地导数 + 多路径梯度累加。** 三个成分各自缺失时的 bug 症状要能说出来。
2. `+` 分发、`*` 交换、tanh 乘 (1-t²) ——三条局部规则覆盖了本视频全部运算。
3. 训练循环四步及顺序:forward → zero_grad → backward → update;zero_grad 的存在是 `+=` 设计的直接后果(两个知识点是同一件事的两面)。
4. Value ≈ 标量 tensor;PyTorch 无新概念,只有新规模。
5. ★ 闭卷重构任务(隔 2‒3 天):空白文件写出 Value(含 backward 拓扑排序与 += )+ Neuron/Layer/MLP + 训练循环,在 4 样本数据集上跑到 loss < 0.05。

**易错点清单**(重构时对照):
- `_backward` 里写成 `=`(共享节点梯度被覆盖)
- 忘记 `self.grad = 1.0` 种子
- 拓扑排序忘了 `reversed`
- 训练循环忘 zero_grad
- `sum(...)` 起始值忘了传 `self.b`(Neuron 里)或默认 int 0 与 Value 相加未处理
