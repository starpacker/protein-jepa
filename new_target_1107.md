实验 E0：统一评测 v3 和 Correct-framework
(这里如果有线程的结果， 可以直接放进来。)
| 模型 | RMSF-r ↑ | Contact J ↑ | ACF Error ↓ | Path Error ↓ | Latent Stable |
|---|---:|---:|---:|---:|---|
| v3 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Correct H+S | 待测 | 待测 | 待测 | 待测 | 待测 |
| Correct H-only | 待测 | 待测 | 待测 | 待测 | 待测 |
| Persistence | 待测 | 待测 | 待测 | 待测 | 不适用 |
| MD Oracle | 参考 | 参考 | 参考 | 参考 | 不适用 |

实验 E1：Representation Sufficiency Probe
这是决定是否继续 Kinetic-JEPA 的第一个关键实验。

我们要回答：
当前 JEPA 表示究竟有没有丢失对未来动力学有用的信息？

暂时不训练新的 propagator。

使用已有 Stage A encoder，构建三种输入：
| Probe | 输入 | 目的 |
|---|---|---|
| P1 | 当前单帧 \(H_t\) | 测量现有 JEPA 状态的信息 |
| P2 | 历史 latent \(H_{t-K:t}\) | 测量额外历史包含的信息 |
| P3 | 当前联合状态 \((H_t,S_t)\) | 测量 S 的实际信息贡献 |

建议选择三类目标：
未来目标	时间跨度
未来 CA 内部距离变化	1、4、16 ns
未来接触状态变化	1、4、16 ns
未来低维集体运动特征	1、4、16 ns


对于每一种目标，使用相同的 proper scoring rule 评价条件概率预测。

定义：
\[
G_\tau^{\mathrm{history}}
=
\mathcal R(P1)-\mathcal R(P2)
\]
如果 \(G_\tau^{\mathrm{history}}>0\)，说明完整历史提供了当前单帧 H 没有保留的额外预测信息。

实验 E2：Frozen JEPA + Kinetic Supervision
这是决定新方法是否有价值的第二个关键实验。

先固定 JEPA，暂时只训练 corrected H/S propagator。

目的非常明确：
仅仅改变动力学训练目标，是否已经足以纠正当前的时间尺度失真？

我建议先实现一个最小的、具有明确时间顺序的训练目标，而不是一次加入 ACF、MSM、VAMP、Closure 和多个物理损失。

具体可以采用 multi-lag joint observable score。

从一条生成轨迹提取：
\[
\Phi_{\mathrm{gen}}
=
\left[
\phi(\hat X_{t+1}),
\phi(\hat X_{t+4}),
\phi(\hat X_{t+16})
\right]
\]
从同一真实 MD 轨迹提取：
\[
\Phi_{\mathrm{MD}}
=
\left[
\phi(X_{t+1}),
\phi(X_{t+4}),
\phi(X_{t+16})
\right]
\]
其中 \(\phi\) 可以先使用一组固定的、可微分的几何动力学特征，例如内部距离、非局部软接触和全局构象尺度。

然后对整个多时间点向量使用 Energy Score：
\[
\boxed{
\mathcal L_{\mathrm{temporal}}
=
\operatorname{ES}
\left(
\{\Phi_{\mathrm{gen}}^{(m)}\}_{m=1}^{M},
\Phi_{\mathrm{MD}}
\right)
}
\]
这个目标的重点是：联合评价同一条轨迹在多个时间点的状态，而不是分别计算每个时间点的无序系综误差。

实验 E3：
1. 键长、键角的合理性，看一下pdb（由人类来看， 不需要你看），键长、键角在合理范围内的百分比
2. V3 和 current model的温度泛化， 需要进行测评，之前应该是没有进行过的。整理一份测评脚本用到的蛋白，以及用到的温度。 我会去跑一份md，去生成测评蛋白在新的温度下的演化路径来测评。 
3. bioemu 作为新的baseline 进行部署， 以及zero-shot 测评， 具体可以参考bioemu2 的测评。
GitHub repo在这里： /liuyutian/protein_jepa_frame/bioemu-main.zip


FIRST OF ALL： 重新确认一下current model 是否合理实现，我们的current model 是在V3 基础上，将S 利用历史信息正确初始化， 但是现在效果居然不如V3了，我需要你检查代码确认current model 被正确实现。 