+++
author = "sh"
title = "record"
date = "2024-7-10"
description = "Today's learning "
categories = [
    "Today's learning"
]
tags = [
    "Today's learning"
]

+++

![](2.jpg)

# 记录

1.修改点积公式尝试，调公式得最优高个4，几就够了，不需要多。



2，PGD 的方法尝试，动静态补丁，总的实验，多追踪两个在otb ，数据集3个，目前就做个transt，训练先单一的，vot 和got 的评估带啊吗要改，不同大小在t，attention 训练 和非训练，直接在原始的放个图，用基地做一个，然后点积的做个表，说明有提升，最后加个长图的演示，补丁位置可以看加上个便宜中心的。

ation 和补丁训练的区别

不同的加权补丁这个可以做一下，以及pgward

cnn 到transformer的过渡和 transformer的

单个layer的，最大的

缩小的

cls 单个 的

softmax后的值有用吗

corss 和self的

25 ，30，35，40，45，50 patch_size

batch_token

书写：

1. 注意力机制的，attention的， 不同追踪器的写一下方法



今日实现：

1。max





1.random

2.追踪的点积鲁棒性

作为一块，单一的尝试和多层的尝试





是个思路

这个方法通过对抗性优化将补丁区域的特征“推离”正常特征空间，影响全局特征提取。下面是将补丁特征与其他区域的特征投影距离最大化的代码示例：import torch.nn.functional as F

# 投影函数
def projection(features):
    # 可以使用PCA、神经网络或者其他方式进行投影
    return F.normalize(features, dim=1)

# 定义自定义损失函数，将补丁区域的特征推离正常特征空间
def feature_projection_loss(model, x, p_x, p_y, p_size):
    features = model.extract_features(x)
    
    # 获取补丁区域的特征并投影
    patch_features = features[:, :, p_x:p_x + p_size, p_y:p_y + p_size]
    projected_patch_features = projection(patch_features)
    
    # 获取其他正常区域的特征并投影
    normal_features = torch.cat([
        features[:, :, :p_x, :], 
        features[:, :, p_x + p_size:, :]
    ], dim=2)
    projected_normal_features = projection(normal_features)
    
    # 最大化补丁与正常特征的投影距离
    loss = -F.mse_loss(projected_patch_features, projected_normal_features)
    return loss

# 优化补丁的对抗扰动
def optimize_patch_projection(model, x, p_x, p_y, p_size, patch_value, steps=100, lr=0.01):
    patch_value = torch.autograd.Variable(patch_value, requires_grad=True)
    optimizer = torch.optim.Adam([patch_value], lr=lr)
    
    for step in range(steps):
        x_adv = add_adversarial_patch(x.clone(), p_x, p_y, p_size, patch_value)
        loss = feature_projection_loss(model, x_adv, p_x, p_y, p_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return patch_value

# 示例用法同上
optimized_patch = optimize_patch_projection(model, x, p_x, p_y, p_size, patch_value)







1,又一个新的思想了，通过cross ，仅仅管最后一层的融合输出，只需要使得补丁位置对应token值下降，然后其他的升高，这也是个思路，主要的思想就是不需要对准p了，开始的需要在意，因为无法更改，后面的应该是能更改到的

2.其实我的就是多头单层，多头多层为什么没有用，看来只有输入的时候最有用。 这就有个猜测了，在第一帧使用cross 能否更近一步。这个可以后面再加。如果有用就证明了只有层才是关键，对于追踪而言开始这层是最有用的，还有mixformer的话，不一样的也可以测试。

3.想到了，怎么使得，没有置信度图的追踪器，用每层叠加的方式确定吗

4，随便一个追踪器的选择的位置，是否依旧有加强，这也是个思想，作为一个分析

5.attention上训练，放在推理为使用attention的会怎么样，能否实现训练上加入attention,也能提升攻击效果

6由置信度图转化成attention map，如果不是置信度的，就使用attention注意力图合并确定最终的补丁位置

7. 分析不同的输入生成补丁是否对结果有影响
8. 固定的补丁大小，应该怎么确定呢
9. 补丁的样子

各个追踪器大小不同，输入的补丁大小应该是多少



最后一层的注意力图还是最后一层的所有注意立图，哪个更好



mix的_prroi_pooling被改了，

head type 里面用center 和 conner ，两种，score_map 是特殊的注意力图，所以使用

仅仅用最后一层的 还是前面综合也需要实现



vot的attention 低于使用的，估计是跑错了，有时间可以在跑个





归一化的问题，我训练的时候经过了双归一化的转换，这就是个问题。之前是因为由不同的归一化，所以需要，现在其实可以以后一个算

补丁对齐的概念？

多数据集共同训练





指标评估这部分得弄好，又需要时间



（1024，1024 ，）

一边就是特征，然后经过v就会变少，所以



k这边成上了v，长度变了，这边得攻击影响更好。

补丁token的转化可以放个公式，这里又设计到了对齐不对齐的问题。

做个cs 的和单个c的就行，好的改，不好的就放在分析里就可以了。只是个任务不用弄的完整，本身就是用来毕业的对吗，工作才是王道，所以我现在需要就是写完75页，然后学习工作需要的东西，在11.20这样的时候就找到个工作，就可以了，加油



把图改成适应追踪的就可以了



不同位置的消融



、

可书写的内容

文章中的**梯度流**是指在点积注意力机制中，梯度在反向传播时通过网络各层的传播方式。在点积注意力机制中，梯度主要流向**值（value token）**，而不是**注意力权重（attention weights）**。这意味着在标准的基于梯度的对抗性攻击（例如PGD攻击）中，模型更容易受到值的影响，而不是注意力机制本身的操作。

具体来说，点积注意力的计算过程如下：

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)VAttention(Q,K,V)=softmax(dkQKT)V

其中，$Q$ 是查询（queries），$K$ 是键（keys），$V$ 是值（values），$d_k$ 是键和查询的维度。注意力的核心在于通过查询与键的点积来计算相似度，然后使用softmax函数归一化这些相似度，最后乘以值的加权和作为输出。

在这种机制下，攻击者通常希望通过改变输入图像中的某些像素或区域来生成对抗性补丁，使得模型的输出发生错误。通过反向传播（backpropagation），攻击者可以计算输入对输出的梯度并调整输入以最小化损失。然而，文章指出，梯度的主要流向是通过值（$V$），而不是通过注意力权重（即由查询与键之间的点积决定的部分）(Lovisotto_Give_Me_Your_…)。

这就意味着**对抗性攻击**如果仅依赖于标准的梯度计算，可能会对模型中的值（$V$）产生更大的影响，而很难直接通过改变查询和键的相似性来操纵注意力机制。这就是为什么作者在提出的**Attention-Fool**攻击中，建议直接优化**预softmax的点积相似度**，以更有效地操控查询和键之间的关系，从而影响模型的注意力分配。

总结来说，梯度流的讨论说明了：**点积注意力机制中，标准梯度攻击往往无法有效修改注意力权重**，这是攻击注意力机制的一大挑战，进而强调了通过修改键（$K$）来操控注意力是一种更有效的攻击方法。



单独q和c，还有很多需要思考，这两天实现就好，有结果就行，今天要把这个attention 给弄完，在分析的时候可以把敏感度模块那部分给弄好，

重点就变了，如果要和别人不一样的话，分析的就是四个部位，实现攻击的不同，加上attention的结构，这个才是重点，已经点积结构的有用，如果这部分有用可能所有的都要冲泡。

如果这两天能弄好就弄，毕竟论文不需要这么用心，重要的还是工作的能力。

选取哪份的输出是个大问题，现在的两个问题还没解决，很麻烦

conner head 是怎么实现确定位置的



不同位置的选择，在注意力里面

如果template to search 影响了攻击，是不是意味着，对他的攻击效果并不好，从这个角度来看的话，在加入攻击的是时候应该只对，search to template



找最高点的方法，可以用公式表示出来

用最后的不合适，应该需要叠加

由最大点定位token的方法这个可以写出来



MixFormer的可视化是个点





定的几个实验，就是q, k, c1,c2 ,qk, c1k,c2k,c1c2,c1c2k



大小，则是25，35，45，55，65



位置 atten，0 ，20，40 ，60 



torch max，torch mean ,torch lay



归一化写一下的公式和laymax 的写下



查个重，还得



少个双c





作为键值更具备攻击效果
