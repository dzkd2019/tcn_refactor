
## 总体判断

这段代码生成的是一套**具有传感器外观的参数化合成数据**，而不是经过实验标定的 PdCu 传感器数字孪生。

它可以做到：

* 模拟基线、漂移、噪声、温湿度影响；
* 生成上升、恢复、延迟、双时间常数等典型曲线；
* 用于检查数据管线、TCN 代码、窗口切分和损失函数；
* 用于预训练、消融实验和鲁棒性测试。

但目前**不能据此证明算法在真实 PdCu 传感器上有效**。它只能反映“设计者假设的传感器响应”，而不是“实测 PdCu 传感器响应分布”。

我的评价是：

| 维度           |  评价 |
| ------------ | --: |
| 软件测试价值       |   高 |
| 曲线形态合理性      |  中等 |
| 统计真实性        |  较低 |
| PdCu 器件专属性   |  很低 |
| 可直接用于最终模型评估  | 不可以 |
| 经实测标定后作为增强数据 |  可以 |

---

# 一、代码具体如何生成数据

## 1. 为每个场景创建独立随机种子

```python
rng = np.random.default_rng(seed)
scenario_seed = int(rng.integers(0, 2**31 - 1))
local = np.random.default_rng(scenario_seed)
```

顶层 `seed` 决定整个数据集，每个场景又有自己的 `scenario_seed`。

优点是：

* 可复现；
* 场景间随机数相互独立；
* 保存 `scenario_seed` 后，可以重新生成某个问题场景。

这是合理的工程设计。

---

## 2. 生成时间轴

```python
length = round(duration_s * config.fs_hz)
time = np.arange(length) / config.fs_hz
```

若：

```python
duration_s = 180
fs_hz = 10
```

则每个场景有：

[
N=180\times10=1800
]

个采样点。

---

## 3. 生成温度和湿度

```python
t0, rh0 = local.uniform(20, 80), local.uniform(0, 80)

temperature = t0 + 1.5 * np.sin(2 * np.pi * time / duration_s)

humidity = rh0 + 2.0 * np.sin(
    2 * np.pi * time / (duration_s * 0.7) + 0.5
)
```

每个场景先随机选择基础环境：

[
T_0\sim U(20,80)
]

[
RH_0\sim U(0,80)
]

然后叠加缓慢正弦变化：

[
T(t)=T_0+1.5\sin(\cdots)
]

[
RH(t)=RH_0+2\sin(\cdots)
]

因此整个数据集大致覆盖：

[
T\in[18.5,81.5]^\circ C
]

[
RH\in[-2,82]%
]

这里已经有一个明显问题：湿度可能小于 0，应当裁剪到：

```python
humidity = np.clip(humidity, 0.0, 100.0)
```

此外，温度和相对湿度被独立生成。真实环境中，如果绝对含水量不变，温度变化会影响相对湿度，因此二者通常不是完全独立的正弦信号。

---

# 二、基线如何生成

## 1. 温湿度相关的标称基线

```python
base_nominal = (
    1.0
    + 0.001 * (temperature - 20)
    - 0.0008 * humidity
)
```

对应：

[
B_{\mathrm{nominal}}(t)
=======================

1
+0.001[T(t)-20]
-0.0008RH(t)
]

含义是：

* 温度每升高 (1^\circ C)，基线增加 1 mV；
* 湿度每增加 1%RH，基线降低 0.8 mV；
* 20°C、0%RH 时基线为 1 V。

这是一个人为指定的线性模型，不是从 PdCu 实验中得到的。

它隐含了很强的假设：

1. 温度对基线始终是正向影响；
2. 湿度对基线始终是负向影响；
3. 温湿度之间没有交互项；
4. 20–80°C 范围内始终线性；
5. 不同器件拥有完全相同的温湿度系数。

真实器件至少应考虑器件级随机系数：

[
B_i(T,RH)=b_{0,i}+b_{T,i}(T-T_{\mathrm{ref}})
+b_{RH,i}(RH-RH_{\mathrm{ref}})
+b_{TRH,i}(T-T_{\mathrm{ref}})(RH-RH_{\mathrm{ref}})
]

---

## 2. 随机游走漂移

```python
drift = np.cumsum(local.normal(0.0, 2e-5, length))
drift -= drift.mean()
drift *= 0.01 / max(float(np.ptp(drift)), 1e-8)
```

首先生成随机游走：

[
d_t=d_{t-1}+\epsilon_t
]

其中：

[
\epsilon_t\sim N(0,2\times10^{-5})
]

然后做两件事：

```python
drift -= drift.mean()
```

使整段漂移均值为 0。

```python
drift *= 0.01 / ptp(drift)
```

把每个场景的漂移峰峰值强制缩放为：

[
\max(d)-\min(d)=0.01\ \mathrm V
]

也就是说，**无论场景时长、采样率和随机轨迹如何，每个场景的漂移峰峰值几乎都是固定的 10 mV。**

这会形成明显的人工统计特征。

更重要的是：

```python
drift.mean()
np.ptp(drift)
```

都使用了整条未来序列。作为数据生成本身不构成直接泄漏，但这不是自然发生的在线漂移过程。

更合理的方法是让漂移强度由扩散系数或每单位时间方差控制，而不是生成后强行统一峰峰值：

[
d_t=d_{t-1}+\sigma_d\sqrt{\Delta t}\epsilon_t
]

还可以叠加慢指数老化、循环漂移和氢暴露导致的不可逆偏置。

真实 Pd 系传感器的长期稳定性会受到表面污染、重复吸放氢和材料结构变化影响；实验研究已经观察到环境污染造成响应与响应时间退化，以及吸放氢循环带来的基线或灵敏度变化。([Nature][1])

最终基线为：

```python
baseline = base_nominal + drift
```

即：

[
B(t)=B_{\mathrm{nominal}}(t)+d(t)
]

---

# 三、气体事件如何生成

## 1. 第一次暴露开始时间固定

```python
cursor = max(
    config.calibration_samples + 100,
    round(25 * config.fs_hz)
)
```

第一段氢气暴露不是随机开始，而是固定从 `cursor` 开始。

默认：

```python
events_per_scenario = 1
```

意味着绝大多数场景中，氢气事件都在相同采样位置开始。

这是代码中最严重的问题之一。

例如，如果第一段暴露总是在第 25 秒开始，模型可能学习：

> 25 秒前浓度为 0，25 秒后大概率有氢气。

尤其你还有“归一化响应时间”特征，模型更容易利用位置而不是传感器动力学。

建议改为：

```python
start = local.integers(min_start, max_start)
```

并让：

* 暴露起点随机；
* 空白段长度随机；
* 部分场景完全没有氢气；
* 部分场景从一开始就存在氢气；
* 部分场景在窗口外开始暴露；
* 训练窗口随机截取，不与事件边界对齐。

---

## 2. 暴露时间和恢复间隔

```python
on = int(local.integers(35, 56) * config.fs_hz)
off = int(local.integers(20, 36) * config.fs_hz)
```

暴露时间：

[
t_{\mathrm{on}}\in[35,55]\ \mathrm s
]

恢复间隔：

[
t_{\mathrm{off}}\in[20,35]\ \mathrm s
]

这里 `integers` 上界不包含，所以分别是 35–55 秒和 20–35 秒。

所有事件都落在较窄的持续时间范围中。真实测试应该包含：

* 很短的脉冲；
* 正常持续暴露；
* 长时间平台；
* 未完全恢复就再次通气；
* 浓度连续变化或阶梯变化。

---

## 3. 浓度

```python
ppm = float(local.uniform(150, 1100))
```

即：

[
C\sim U(150,1100)\ \mathrm{ppm}
]

这意味着浓度是连续均匀分布，而实验标定通常是离散浓度点，例如：

[
0,100,200,400,600,800,1000,1200\ \mathrm{ppm}
]

连续均匀分布可用于数据增强，但不能代替实际标定点。

---

# 四、响应幅度如何生成

## 1. 温湿度修正后的“灵敏度”

```python
sensitivity = 0.0015 * (
    1 + 0.005 * (temperature[start] - 20)
) * (
    1 - 0.006 * humidity[start]
)
```

对应：

\[
S(T,RH)=0.0015[1+0.005(T-20)][1-0.006RH]
\]

它规定：

* 温度越高，灵敏度越高；
* 湿度越高，灵敏度越低；
* 温湿度作用相乘；
* 整个暴露期间使用事件开始时的温湿度。

最后一点很重要。即使暴露过程中温湿度在变化：

```python
temperature[start:end]
humidity[start:end]
```

响应参数仍只取：

```python
temperature[start]
humidity[start]
```

这与代码中动态变化的温湿度不完全一致。

Pd 基传感器确实可能受到湿度影响，水分会占据或污染表面吸附位点并改变响应时间和信号幅度；但具体影响方向和大小依赖材料组成、保护层、载气及工作温度，不能直接假设为固定的线性下降。([Nature][2])

---

## 2. 浓度—幅度关系

```python
amplitude = sensitivity * ppm**0.85
```

即：

[
A=S(T,RH)C^{0.85}
]

这表示响应幅度随浓度次线性增加。

但指数 `0.85` 是硬编码的，没有实验依据。

此外，从量纲上更规范的写法应当是：

[
A=A_{\mathrm{ref}}
\left(\frac{C}{C_{\mathrm{ref}}}\right)^n
]

而不是直接计算：

[
C^{0.85}
]

否则 `sensitivity` 的单位实际上是：

[
\mathrm{V}/\mathrm{ppm}^{0.85}
]

它已经不是通常意义上的线性灵敏度。

按照当前参数，完整稳态幅度大约覆盖：

[
0.053\sim0.755\ \mathrm V
]

部分典型值为：

|       浓度 | 20°C、0%RH | 20°C、80%RH | 80°C、0%RH |
| -------: | --------: | ---------: | --------: |
|  150 ppm |   0.106 V |    0.055 V |   0.138 V |
|  600 ppm |   0.345 V |    0.179 V |   0.448 V |
| 1100 ppm |   0.577 V |    0.300 V |   0.750 V |

这相对于约 1 V 的基线是一个很大的变化。是否合理完全取决于：

* 传感器是恒流还是恒压驱动；
* 输出是电阻、电桥电压还是放大后电压；
* 放大器增益；
* PdCu 薄膜厚度和结构；
* Cu 含量；
* 实际器件灵敏度。

必须与真实数据比较。

更关键的是，PdCu 体系的浓度关系不一定是固定 (C^{0.85})。一个使用 Pd(*{75})Cu(*{10})Si(_{15}) 薄膜的实验研究发现，其吸氢量更符合关于 (\sqrt C) 的 Langmuir 型饱和关系，而低占据率区域才近似平方根规律。该器件和你的电阻型 PdCu 器件不完全相同，因此不能直接套用，但足以说明指数 0.85 不是 PdCu 的通用定律。([Nature][1])

---

# 五、时间常数如何生成

```python
tau = 30.0 * (
    1 + 0.020 * humidity[start]
) / np.exp(
    0.012 * (temperature[start] - 20)
)
```

即：

[
\tau(T,RH)
==========

30
\frac{1+0.02RH}
{\exp[0.012(T-20)]}
]

它规定：

* 湿度升高，响应变慢；
* 温度升高，响应加快；
* 20°C、0%RH 时 (\tau=30) 秒。

然后加入浓度影响：

```python
tau *= (ppm / 600.0) ** local.uniform(-0.18, 0.18)
```

即：

[
\tau'
=====

\tau
\left(\frac{C}{600}\right)^q
]

其中每个事件重新随机：

[
q\sim U(-0.18,0.18)
]

这意味着在某个事件中：

* 浓度越高，响应可能越快；

而在另一个事件中：

* 浓度越高，响应又可能越慢。

这种做法增加了数据多样性，却缺少明确物理意义。对于同一种器件，浓度与时间常数的关系通常应具有稳定趋势，或者至少由器件级参数控制，而不是每次暴露随机改变正负方向。

在当前参数范围内，响应时间常数大约可达到：

[
\tau\approx11\sim103\ \mathrm s
]

恢复时间常数还会扩大到大约：

[
\tau_{\mathrm{down}}
\approx9\sim207\ \mathrm s
]

而氢气暴露只有 35–55 秒。因此很多事件在停止通气时还远未达到定义的 `amplitude`。

例如单指数响应在 (t=\tau/2) 时只达到：

[
1-e^{-0.5}\approx39.3%
]

这可以用于模拟“加速预测”任务，但意味着模型大部分时间只能依赖瞬态斜率推断浓度。

---

# 六、四种动力学如何生成

代码随机选择：

```python
kinetics = local.choice(
    ["single", "double", "stretched", "delayed"]
)
```

四种模型概率相等，均为 25%。

## 1. 单指数

```python
progress = 1.0 - np.exp(-event_time / tau)
```

对应：

[
p(t)=1-e^{-t/\tau}
]

响应为：

[
r(t)=A,p(t)
]

这是标准一阶响应。

---

## 2. 双指数

```python
progress = 1.0 - (
    mix * np.exp(-event_time / tau_fast)
    + (1.0 - mix) * np.exp(-event_time / tau_slow)
)
```

对应：

[
p(t)
====

1-ae^{-t/\tau_f}
-(1-a)e^{-t/\tau_s}
]

其中：

[
a\in[0.25,0.75]
]

[
\tau_f\in[0.25\tau,0.65\tau]
]

[
\tau_s\in[1.3\tau,2.5\tau]
]

它可以近似表达：

* 表面吸附和体扩散；
* 快慢晶粒区域；
* 气室交换与材料响应；
* 两种吸氢位点。

在概念上比单指数更真实。

实验中 Pd 合金吸氢可以表现为表面解离吸附和体相吸收两个耦合过程，因此用多状态或双时间常数模型具有物理合理性。([Nature][1])

---

## 3. 拉伸指数

```python
beta = local.uniform(0.55, 1.45)

progress = 1.0 - np.exp(
    -np.power(event_time / tau, beta)
)
```

对应：

[
p(t)
====

1-\exp\left[-(t/\tau)^\beta\right]
]

* (\beta=1)：单指数；
* (\beta<1)：长尾、分布式时间常数；
* (\beta>1)：前期较平缓，之后加速。

但这里有一个数值和物理风险。

当：

[
0<\beta<1
]

其理论初始导数为：

[
\frac{dp}{dt}
=============

\frac{\beta}{\tau}
\left(\frac t\tau\right)^{\beta-1}
e^{-(t/\tau)^\beta}
]

当 (t\rightarrow0) 时可能趋于无穷大。离散采样不会真的产生无穷值，但在高采样率下可能产生过陡的首段响应。

应当加入气室传输或有限表面速率，避免无限初始斜率。

---

## 4. 延迟一阶响应

```python
delay_s = local.uniform(0.2, 2.0)

effective_time = np.maximum(
    event_time - delay_s, 0.0
)

progress = 1.0 - np.exp(
    -effective_time / tau
)
```

对应带纯滞后的模型：

[
p(t)=
\begin{cases}
0,&t<t_d\
1-e^{-(t-t_d)/\tau},&t\ge t_d
\end{cases}
]

它可以近似表示：

* 阀门切换延迟；
* 管路输运延迟；
* 气室混合延迟；
* 传感膜启动滞后。

但是代码只延迟了传感器响应，标签中的真实浓度仍然瞬间从 0 跳到 `ppm`。这相当于假设气体已瞬间到达传感器，但传感器等待一段时间后才响应。

真实系统中通常应该区分：

[
C_{\mathrm{command}}
]

MFC 或阀门设定浓度，以及：

[
C_{\mathrm{sensor}}(t)
]

真正到达传感器表面的浓度。

---

# 七、暴露结束后的恢复

```python
tau_down = tau * local.uniform(0.8, 2.0)

response[end:] += (
    amplitude
    * progress[-1]
    * np.exp(-recovery_time / tau_down)
)
```

对应：

[
r(t)
====

r(t_{\mathrm{end}})
e^{-(t-t_{\mathrm{end}})/\tau_{\mathrm{down}}}
]

其中：

[
\tau_{\mathrm{down}}\in[0.8\tau,2\tau]
]

这是代码中较合理的设计：

* 响应和恢复使用不同时间常数；
* 暴露停止时信号连续；
* 不要求暴露阶段已经达到稳态；
* 可以表现不完全恢复。

PdCuSi 实验中也观察到释放过程可能比吸收过程慢，并且首次循环与后续循环之间可能存在差异。([Nature][1])

不过这里仍有两个问题。

## 问题 1：恢复只有单指数

无论上升阶段是：

* 双指数；
* 拉伸指数；
* 延迟响应；

恢复阶段一律变成单指数。

真实吸氢和脱氢可能具有不同的模型阶次，而不只是不同的时间常数。

## 问题 2：多个事件直接相加

```python
response[start:end] += ...
response[end:] += ...
```

如果有多个事件，旧事件的恢复尾部和新事件响应直接叠加。

在线性、小信号区，这可以看作脉冲响应叠加；但在以下情况下不再合理：

* 高浓度接近饱和；
* 氢占据位点有限；
* 存在滞后；
* 前次氢未释放完；
* 灵敏度随吸氢状态变化。

直接相加可能使响应超过该浓度下可能达到的物理平衡值。

更合理的是维护单一传感器状态：

[
\frac{dy}{dt}
=============

\frac{F(C,T,RH)-y}{\tau(C,T,RH,m)}
]

而不是为每个事件独立生成一个波形后求和。

---

# 八、浓度标签如何生成

```python
concentration[start:end] = ppm
```

标签是理想矩形脉冲：

[
C(t)=
\begin{cases}
C_0,&t_{\mathrm{start}}\le t<t_{\mathrm{end}}\
0,&\text{其他}
\end{cases}
]

这表示气体浓度在一个采样周期内瞬间从 0 跳到目标浓度，并在结束时瞬间变回 0。

对于 MFC、管道和测试腔组成的真实系统，更合理的模型是：

[
\tau_{\mathrm{chamber}}
\frac{dC_s}{dt}
+
C_s
===

C_{\mathrm{command}}(t-t_d)
]

然后让传感器响应 (C_s(t))，而不是理想命令值。

否则模型实际上同时学习了：

1. 气路传输；
2. 传感器动力学；

但标签却把气路延迟忽略了。

---

# 九、最终电压和噪声

```python
voltage = baseline + response

voltage += local.normal(
    0.0, 0.004, length
)
```

最终：

[
V(t)=B(t)+r(t)+n(t)
]

其中：

[
n(t)\sim N(0,0.004^2)
]

也就是标准差 4 mV 的独立白噪声。

这种噪声模型过于理想。真实系统还可能存在：

* 低频 (1/f) 噪声；
* ADC 量化；
* 工频及电磁干扰；
* 放大器偏置和温漂；
* 电源纹波；
* 流量切换尖峰；
* 突发异常值；
* 与信号幅度相关的异方差噪声；
* 温湿度传感器自身噪声；
* 不同通道之间的相关噪声。

并且现在：

```python
temperature
humidity
```

完全没有测量误差，而只有 `voltage` 有噪声。这会让温湿度补偿显得比真实系统容易。

---

# 十、代码中做得比较好的部分

## 1. 没有预设所有响应都是单指数

随机加入：

* 单指数；
* 双指数；
* 拉伸指数；
* 延迟一阶。

这比只生成一种一阶曲线更适合测试 TCN 是否过度依赖固定模型。

## 2. 上升和恢复不对称

独立的 `tau_down` 是合理设计。

## 3. 保留残余响应

气体结束后信号不会瞬间回到基线，这一点比理想方波真实。

## 4. 明确保存事件起止位置

```python
event_start
event_end
event_ppm
```

事件标签完整，不再需要根据浓度数组猜测暴露结束位置。

## 5. 复现性良好

每个场景都有独立种子。

## 6. NPZ 不使用 pickle

```python
allow_pickle=False
```

安全性较好，也避免依赖 Python 对象反序列化。

## 7. 解压逻辑合理

在循环外读取：

```python
voltage = data["voltage"]
```

避免在每个场景循环中重复解压同一数组。

---

# 十一、最严重的可靠性问题

## 1. 所有核心物理参数都是人为设定的

包括：

```python
0.001
-0.0008
0.0015
0.005
0.006
0.85
30.0
0.020
0.012
```

这些参数没有来自你的 PdCu 器件实验。

因此，该生成器无法回答：

* 你的器件灵敏度是多少；
* 浓度指数是否为 0.85；
* 湿度影响是否为每 %RH 降低 0.6%；
* 温度是否总是提高灵敏度；
* 时间常数是否为 30 秒；
* 温度激活系数是否为 0.012；
* 恢复时间是否为响应时间的 0.8–2 倍。

它只是把这些结论写进了训练数据。

---

## 2. 第一段事件位置固定，存在时间泄漏

默认一个事件时，几乎所有场景都在同一位置开始暴露。

如果网络能感知序列位置，测试结果可能严重虚高。

应优先修复。

---

## 3. “物理粗估特征”可能形成循环论证

假设你使用同一个生成器：

[
A\propto C^{0.85}
]

并在特征 6 中使用近似相同的方程反演浓度，那么：

* 训练数据由该方程生成；
* 输入特征又由该方程反演；
* 测试数据仍由同一方程生成。

模型获得高准确率，只能证明它成功反演了生成器，而不能证明其适用于真实 PdCu 传感器。

合成数据测试必须额外加入：

* 方程参数失配；
* 不同浓度规律；
* 未见过的动力学；
* 未见过的漂移和噪声；
* 真实实验数据。

---

## 4. `true_baseline` 是不可观测的理想信息

```python
baseline.astype(np.float32)
```

被完整保存。

如果训练或推理特征直接使用这个 `true_baseline`，就等于给模型提供了真实世界中通常无法获取的 oracle 信息。

真实部署时只能使用：

* 启动校准得到的基线；
* 在线基线估计；
* 无氢状态下的基线更新；
* 温湿度补偿后的预测基线。

训练时应严格区分：

```python
true_baseline       # 只用于评估
estimated_baseline  # 才能用于模型输入
```

---

## 5. 每次事件随机切换动力学类型

同一传感器的第一个事件可能是单指数，第二个事件可能突然变成拉伸指数，第三个又变成纯延迟一阶。

真实器件的动力学可能随条件变化，但通常具有器件级和结构级连续性。

更合理的层次模型是：

```text
器件级参数：
    动力学类别
    薄膜灵敏度
    基础时间常数
    温度系数
    湿度系数
    老化速度

场景级参数：
    当前基线
    当前污染程度
    环境条件

事件级参数：
    小幅重复性波动
    阀门延迟
    流量误差
```

而不是每个事件完全重新抽取物理机制。

---

## 6. 漂移分布存在明显人工痕迹

所有场景漂移峰峰值固定为 10 mV。

模型可能学到这种固定统计特征，而且无法覆盖：

* 漂移很小的器件；
* 漂移很大的老化器件；
* 单向漂移；
* 暴露诱导漂移；
* 突然基线跳变。

---

## 7. 没有老化、污染和首次循环效应

代码中每次事件的幅度不会因为历史暴露而系统性下降，也没有：

* 首次吸氢调理；
* 循环后的不可逆结构变化；
* CO、CO₂、H₂S 等污染；
* 薄膜开裂或应力松弛；
* 长期灵敏度衰减。

实验研究表明，Pd 合金薄膜可能出现首次循环和后续循环不同、残余氢难以完全清除等现象；Pd 表面污染也可能导致长期响应性能下降。([Nature][1])

---

# 十二、保存和加载代码的问题

保存和加载总体上是合理的，但仍有几个工程问题。

## 1. 配置保存后没有被加载

```python
config=np.asarray(
    [asdict(config)],
    dtype=str
)
```

但 `load_scenarios()` 完全没有读取 `config`。

因此加载后不知道：

* 采样率；
* 窗口长度；
* 标定样本数；
* 数据生成器版本。

如果调用方错误地使用另一个 `SystemConfig`，可能产生静默错误。

建议保存为 JSON：

```python
config_json = json.dumps(
    asdict(config),
    ensure_ascii=False
)
```

并在加载时返回：

```python
return scenarios, config
```

同时保存：

```python
schema_version
generator_version
created_at
```

## 2. 缺少数据校验

加载时应检查：

* 所有数组第一维一致；
* 所有序列长度一致；
* `start < end <= length`；
* `event_ppm` 非负；
* 事件不非法重叠；
* 浓度数组与事件标签一致；
* 无 NaN 和 Inf。

## 3. 两种浓度精度不同

逐点浓度：

```python
float32
```

事件浓度：

```python
float64
```

保存后两者可能有极小差异。因此不能使用严格相等比较：

```python
concentration == event_ppm
```

应使用：

```python
np.isclose(...)
```

---

# 十三、它在什么意义上“像真实传感器”

它可以模拟以下定性特征：

* 有温度和湿度交叉影响；
* 存在基线漂移；
* 响应不是瞬时完成；
* 上升和恢复不对称；
* 部分响应有延迟；
* 可能存在多个时间常数；
* 气体结束后有残留尾部；
* 电压含有测量噪声。

所以，从绘制曲线的角度看，它很可能“看起来像”传感器数据。

但数据看起来像，并不代表联合统计分布真实。模型实际关心的是：

[
P(V_{0:t},T_{0:t},RH_{0:t}\mid C_t,\text{device})
]

当前代码只定义了一个人为的：

[
P_{\mathrm{sim}}
]

而尚未证明：

[
P_{\mathrm{sim}}\approx P_{\mathrm{real}}
]

---

# 十四、如何判断生成数据是否可靠

应当采集真实 PdCu 数据，并至少比较以下指标。

## 1. 稳态响应曲线

比较：

[
A(C,T,RH)
]

检查：

* (C^{0.85}) 是否成立；
* 是否存在饱和；
* 是否更接近平方根、Langmuir 或分段关系；
* 不同温湿度下曲线是否只是简单缩放。

## 2. 动态响应指标

比较真实和模拟数据中的：

[
t_{10},t_{50},t_{63},t_{90}
]

以及：

[
\frac{t_{90}}{t_{50}}
]

如果比值随浓度或环境明显变化，说明单时间常数不足。

## 3. 上升和恢复残差

分别拟合：

* 单指数；
* 双指数；
* 拉伸指数；
* 带纯滞后一阶；
* 两状态吸附—吸收模型。

根据真实拟合频率决定生成器中各模型的概率，而不是固定各 25%。

## 4. 噪声频谱

在无氢、恒温恒湿条件下采集长时间基线，比较：

* 标准差；
* 功率谱密度；
* 自相关；
* Allan 方差；
* 异常点率；
* 量化台阶。

如果真实噪声有明显相关性，就不能只使用独立 Gaussian 白噪声。

## 5. 漂移分布

比较不同器件和不同时间尺度下：

[
\Delta B/\Delta t
]

以及循环前后的基线偏移。

## 6. 分布距离

可以使用：

* Wasserstein 距离；
* Maximum Mean Discrepancy；
* 动态时间规整距离；
* 特征分布 KL/JS 距离；
* 训练“真实/合成”二分类器。

如果一个简单分类器能轻易区分真实与合成数据，说明生成器还不够逼真。

---

# 十五、建议的改造方向

## 第一优先级：消除标签泄漏

至少修改为：

```python
first_start = int(
    local.uniform(min_start_s, max_start_s)
    * config.fs_hz
)
cursor = first_start
```

并加入无事件场景：

```python
if local.random() < blank_probability:
    events_for_this_scenario = 0
```

不要让训练窗口总是从场景第 0 点开始，也不要总是与暴露起点对齐。

---

## 第二优先级：维护统一的传感器状态

先生成真实到达传感器表面的浓度：

[
\frac{dC_s}{dt}
===============

\frac{C_{\mathrm{command}}-C_s}
{\tau_{\mathrm{chamber}}}
]

再生成传感器状态：

[
\frac{dy}{dt}
=============

\frac{F(C_s,T,RH)-y}
{\tau(C_s,T,RH,m)}
]

离散形式为：

```python
c_sensor[t] = (
    c_sensor[t - 1]
    + dt / tau_chamber
    * (c_command[t - delay] - c_sensor[t - 1])
)

target = equilibrium_response(
    c_sensor[t], temperature[t], humidity[t]
)

response[t] = (
    response[t - 1]
    + dt / tau
    * (target - response[t - 1])
)
```

这样可以自然处理：

* 多次事件；
* 不完全恢复；
* 浓度改变；
* 饱和；
* 历史状态；
* 非矩形气体输入。

---

## 第三优先级：从实验标定幅度函数

不要固定：

```python
ppm**0.85
```

优先考虑由实验数据拟合：

[
F(C,T,RH)
]

可以使用：

* 单调 PCHIP；
* 单调样条；
* Langmuir 型函数；
* 基于 (\sqrt C) 的饱和函数；
* 分温湿度查表；
* 带器件随机效应的层次模型。

例如：

[
F(C,T,RH)
=========

A_{\max}(T,RH)
\frac{K(T,RH)\sqrt C}
{1+K(T,RH)\sqrt C}
]

这只是候选模型，仍必须由你的器件数据验证。PdCuSi 的实测研究支持类似的平方根浓度 Langmuir 型关系，但传感机理和器件结构不同，不能直接无标定移植。([Nature][1])

---

## 第四优先级：建立器件级随机参数

每个场景最好属于某个虚拟器件：

```python
device = sample_device_parameters()
```

器件参数在所有事件中保持稳定：

```python
device.sensitivity_scale
device.tau_scale
device.temp_coefficient
device.humidity_coefficient
device.kinetics_type
device.noise_level
device.drift_rate
device.aging_rate
```

事件之间只加入小幅随机变化。

这样才能模拟真实的：

* 器件间差异；
* 同一器件重复性；
* 跨器件泛化。

---

## 第五优先级：加入真实噪声和故障

建议至少加入：

[
n_t=n_{\mathrm{white}}+n_{\mathrm{AR}}+n_{1/f}
+n_{\mathrm{spike}}+n_{\mathrm{quantization}}
]

同时为温度和湿度加入：

* 测量噪声；
* 时间延迟；
* 量化；
* 偶发缺失值。

---

# 最终结论

这段代码的定位应该是：

> 一个用于验证算法流程和构造多样瞬态曲线的合成数据生成器。

而不能定位为：

> 可以代表真实 PdCu 合金氢传感器的高保真数据模型。

它的主要优点是曲线类型丰富、标签完整、可复现，并考虑了温湿度、漂移、恢复和噪声。其主要缺陷是核心参数完全未经实测标定、第一事件起点固定、真实基线可能成为 oracle、浓度响应关系和温湿度关系被硬编码、噪声与老化过于简单，以及多个事件使用独立响应直接叠加。

在当前状态下，适合：

* 单元测试；
* 算法调试；
* TCN 输入输出尺寸验证；
* 预训练；
* 鲁棒性消融。

不适合：

* 汇报真实浓度预测精度；
* 比较算法是否优于传统物理模型；
* 证明加速补偿能够部署；
* 作为论文中唯一的训练和测试数据；
* 估计真实器件的检测限、响应时间和误差。

最关键的原则是：

[
\boxed{
\text{合成数据可以补充真实数据，但不能验证自身的真实性}
}
]

最终性能必须在完全独立的真实 PdCu 器件数据上评估，最好采用“留一器件验证”和“留一温湿度工况验证”，而不是仅在同一生成器产生的测试集上评估。

[1]: https://www.nature.com/articles/s41598-021-98347-4 "2-step reaction kinetics for hydrogen absorption into bulk material via dissociative adsorption on the surface | Scientific Reports"
[2]: https://www.nature.com/articles/s41467-024-53080-0?utm_source=chatgpt.com "Long-term reliable wireless H2 gas sensor via repeatable ..."
