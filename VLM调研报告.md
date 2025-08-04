---
title: VLM调研报告
date: 2025-08-04 21:00:49
tags:
---



<h1 id="oJnjd">概述</h1>

<font style="color:rgb(25, 27, 31);">计算机视觉（Computer Vision）核心在于通过算法赋予计算机"看懂"图像和视频的能力。传统CV技术体系围绕对视觉内容的</font>**<font style="color:rgb(25, 27, 31);">单点</font>**<font style="color:rgb(25, 27, 31);">理解与分析展开：基础任务如图像分类，通过深度学习方法判断图片主体类别；目标检测实现物体的定位与识别，用矩形框标记位置；语义分割将每个像素归类到特定对象类别，形成精细化的区域划分。应用层技术则包括光学字符识别（OCR）将图片中文字检测并识别出来，以及结合人脸检测、特征提取与比对的人脸识别系统等。</font>



**<font style="color:rgb(25, 27, 31);">基础CV模型能力环环相扣、相互协作，共同构建起从感知到理解的完整视觉智能体系。</font>**<font style="color:rgb(25, 27, 31);">然而，传统的CV模型往往基于卷积神经网络 (CNN)构建，并且是在有限类别的数据集合上针对特定任务进行训练的。模型往往无法超越其训练的任务或类别集。如果测试用例发生根本变化或需要添加新的类别，算法工程师则须收集和标记大量图像并重新训练模型。这是一个昂贵且耗时的过程。此外，CV 模型往往没有任何自然语言理解的能力。</font>



<font style="color:rgb(25, 27, 31);">为解决传统CV模型的泛化瓶颈与语义鸿沟，自监督学习和多模态学习范式逐渐兴起。2020年Google Research提出的</font><font style="color:#DF2A3F;">Vision Transformer（ViT）</font><font style="color:rgb(25, 27, 31);">首次将Transformer架构引入视觉领域，通过</font><font style="color:#DF2A3F;">全局注意力机制建模图像块序列</font><font style="color:rgb(25, 27, 31);">的关联性，突破CNN的局部归纳偏置限制，首次验证了Transformer在视觉任务上的可行性。</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">随后2021年，OpenAI发布CLIP模型（Contrastive Language-Image Pre-training），其整体结构为经典的“双塔结构”：</font><font style="color:#DF2A3F;">视觉编码器采用ViT，文本编码器采用Bert</font><font style="color:rgb(25, 27, 31);">。基于4亿规模图文对，采用对比学习框架对齐视觉与语言模态，实现了图像与文本的跨模态统一对齐，在零样本分类任务中显著超越传统监督模型，验证了"提示词工程"在开放域视觉任务中的可行性。</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">2023年，Salesforce提出的</font><font style="color:#DF2A3F;">BLIP-2进一步推动多模态预训练效率</font><font style="color:rgb(25, 27, 31);">，通过冻结预训练视觉与语言模型参数、引入</font><font style="color:#DF2A3F;">轻量级查询Transformer桥接两模态</font><font style="color:rgb(25, 27, 31);">，显著降低计算成本的</font><font style="color:#DF2A3F;">同时保持强大跨模态对齐能力</font><font style="color:rgb(25, 27, 31);">，为资源受限场景提供高效解决方案。这些技术突破为视觉与语言的多模态深度融合奠定基础，最终催生出能跨越模态边界进行开放式回答的视觉语言模型。</font>



大型语言模型的出现标志着人工智能领域转型的开始，它们在文本信息处理上的能力极大地推动了这一进程。尽管LLMs在文本处理上表现出色，但它们<font style="color:#DF2A3F;">主要限于处理单一模态的数据，即文本。</font>这限制了它们在<font style="color:#DF2A3F;">理解和生成涉及图像和视频等多模态数据</font>方面的能力。为了克服LLMs的局限性，研究人员开发了<font style="color:#DF2A3F;">视觉-语言模型（VLMs）</font>。这些模型结合了视觉和文本信息，展示了在理解和生成涉及图像和文本的内容方面的卓越能力。

<font style="color:rgb(25, 27, 31);">视觉语言模型 (Vision Language Model, VLM) 是通过将大语言模型 (LLM) 与视觉编码器（Vision Encoder）相结合构建的多模态 AI 系统，</font><font style="color:#DF2A3F;">通过将视觉映射到语言，让LLM能够理解和处理视频、图像和文本，使模型具有“看”的能力。</font><font style="color:rgb(25, 27, 31);background-color:#FBDE28;">与传统的CV模型不同，</font><font style="color:#DF2A3F;background-color:#FBDE28;">VLM 不受固定类别集或特定任务 (如分类或检测) 约束</font><font style="color:rgb(25, 27, 31);">。通过在大量文本和图像视频-文本对的语料上进行训练，VLM 可以用自然语言处理许多典型的视觉任务（OCR、图像分类、目标检测、图像分割、人脸识别等）以及新的生成式 AI 任务，例如视频摘要和图像问答等。</font>



<font style="color:rgb(25, 27, 31);">总体来说，视觉语言模型VLM从2023年开始迅猛发展。VLM模型的快速进步，离不开如下4个关键因素：</font>

1. **<font style="color:rgb(25, 27, 31);background-color:#FBDE28;">ViT (Vision Transformer)模型架构的成熟</font>**<font style="color:rgb(25, 27, 31);">：谷歌首次尝试将 Transformer 直接应用于计算机视觉，使得视觉和语言模态之间的特征融合变得更加容易和有效，为后续 VLM 的发展提供了重要的视觉特征提取方法 。</font>
2. **<font style="color:rgb(25, 27, 31);">OpenAI </font>****<font style="color:rgb(25, 27, 31);background-color:#FBDE28;">CLIP</font>****<font style="color:rgb(25, 27, 31);">(Contrastive Language-Image Pretraining)模型的成功</font>**<font style="color:rgb(25, 27, 31);">：基于对比学习将图像和文本映射到一个共同的特征空间中，实现图像和文本之间的相互检索和理解。CLIP 的成功展示了多模态预训练的强大潜力。</font>
3. **<font style="color:rgb(25, 27, 31);background-color:#FBDE28;">生成式自回归语言大模型（LLM）的兴起</font>**<font style="color:rgb(25, 27, 31);">：基于Next Token Predict范式的GPT系列生成式自回归语言大模型在自然语言处理领域取得了巨大成功，展现出强大的语言生成和理解能力。</font>
4. **<font style="color:rgb(25, 27, 31);background-color:#FBDE28;">硬件算力的提升和数据scaling law的广泛验证</font>**<font style="color:rgb(25, 27, 31);">：GPU集群性能的指数级增长使千亿参数的多模态模型训练成为可能，同时数据规模与模型性能的scaling law被广泛验证，极大地推动业界系统性地优化算力分配与数据配比，大幅降低训练成本并加速模型迭代效率。</font>

<font style="color:rgb(25, 27, 31);">四者共同构建了视觉-语言跨模态对齐的技术闭环，使VLM在语义理解、指令跟随等核心能力上实现突破。值得注意的是，VLM模型的兴起并不代表传统CV模型的没落，很多专业CV任务，例如人脸识别，VLM在此类垂直任务下的效果并不能令人满意，往往这类场景还是需要专用的模型才能得到很好的处理。</font>



VLM的应用：

1，图像字幕生成：VLMs可以自动为图像生成描述性文字，这在社交媒体、内容管理和辅助视觉障碍人士方面非常有用。

**2，视觉问答（VQA）**：**VLMs能够理解自然语言问题并根据图像内容提供答案，这项技术可以用于教育软件、虚拟助手和交互式客户服务系统。**

3，图像检索：通过理解和索引图像内容及其相关文本，VLMs可以用于改进图像搜索技术，提供更准确和相关的搜索结果。

4，内容创作：VLMs可以根据给定的文本描述生成图像或视频，这对于艺术创作、游戏设计和电影制作等领域具有创新性的影响。

5，文档理解：帮助翻译多模态内容，例如带有图片说明的文档材料。

6，医疗影像分析：在医疗领域，VLMs可以辅助医生分析医学影像，并结合相关病例报告或临床笔记，支持诊断过程。

7，自动驾驶：VLMs可以整合视觉传感器数据和语言指令，帮助自动驾驶系统更好的决策。

8，增强现实和虚拟现实：在AR和VR应用中，VLMs可以提供对用户视觉和语言输入的更深层次理解，从而创造更丰富的交互体验。

9，社交媒体分析：VLMs可以分析社交媒体上的图像和文本内容，以识别趋势、情感或其他有用的信息。



<h1 id="FkwUp">0 多模态大模型MM-LLM</h1>
**<font style="color:#DF2A3F;">视觉语言模型（Vision language model）属于多模态大模型（multi-modality LLM，MM-LLM）的子集</font>**：在MM-LLM框架下，模型的整体结构往往包含如下图所示的5个部分：

**<font style="color:#DF2A3F;">MM-LLM架构：</font>**

+ **<font style="color:rgb(25, 27, 31);">模态编码器（Modality Encoder）：</font>**<font style="color:rgb(25, 27, 31);">将多模态的数据（文本、语音、图像）编码成向量空间特征，该模块通常是单独进行预训练的，典型的方法有基于CNN的Resnet，基于Transformer的ViT等。</font>
+ **<font style="color:rgb(25, 27, 31);">输入投影层（Input Projector）：</font>**<font style="color:rgb(25, 27, 31);">将模态编码器的</font>**<font style="color:rgb(25, 27, 31);">输出映射到LLM的输入特征空间</font>**<font style="color:rgb(25, 27, 31);">的适配层，一般模型结构比较简单，不同的多模态模型一般是随机初始化该模块的参数做冷启训练。典型的网络层：</font>**<font style="color:#DF2A3F;">MLP，Cross-Attention等。</font>**
+ **<font style="color:rgb(25, 27, 31);">LLM主干网络（LLM Backbone）：</font>**<font style="color:rgb(25, 27, 31);">LLM是经过预训练的模型，一般还要串联多个模块继续做Post-Pretrain和微调，使得模型能识别多模态的特殊token和多模态的特征输入。</font>
+ **<font style="color:rgb(25, 27, 31);">输出投影层（Output Projector）：</font>**<font style="color:rgb(25, 27, 31);">将LLM生成的数据，映射成Modality Generator 可理解的特征空间，</font>**<font style="color:#DF2A3F;">一般是简单的Transformer层或MLP层</font>**<font style="color:rgb(25, 27, 31);">。</font>
+ **<font style="color:rgb(25, 27, 31);">模态生成器（Modality Generator）：</font>**<font style="color:rgb(25, 27, 31);">多模态的生成器，最终输出多模态的结果如图像、语音、视频等。模型基本都是</font>**<font style="color:rgb(25, 27, 31);">基于LDM（Latent Diffusion Models）</font>**<font style="color:rgb(25, 27, 31);">的衍生模型，</font>**<font style="color:rgb(25, 27, 31);">如图片领域的Stable Diffusion方法</font>**<font style="color:rgb(25, 27, 31);">。</font>


![](mm-llm.png)


<font style="color:rgb(25, 27, 31);">模型整体以LLM为核心主干，分别在前后有一个输入、输出的投影模块（Projector），投影模块主要是用于桥接不同模态输入和输出。输入投影模块（Input Projector）用于将模态编码器处理的不同模态特征映射到文本特征空间，以便输入给LLM；输出投影模块（Output Projector）用于将文本特征空间结果映射到模态生成器的输入空间，以引导模态生成器生成多模态结果。</font>



<h1 id="DoT4f">1 VLM：</h1>
<h2 id="FVVKL"><font style="color:#DF2A3F;">VLM基础架构：</font></h2>
+ 由图像和文本编码器生成嵌入
+ 在图像和文本<font style="color:#DF2A3F;">融合层中进行融合</font>
+ 将融合向量通过LLM生成最终的视觉感知生成文本


![](vlm结构.png)


**<font style="color:#DF2A3F;">以LLaVA为例进行分析：</font>**

Large Language and Vision Assistant，LLaVA，作为最早一批出现的视觉语言大模型之一，其结构分为Vision Encoder、Projection和LLM三部分，其模型结构与训练策略对于现在的多模态模型和视觉语言模型的发展产生了巨大的影响，主要包含如下图所示的3个部分。


![](llava.png)

由图中所示，常见的VLM整体架构包含一个 Vision Encoder 模块，负责提取图片特征；一个 Projection，负责将图片特征映射到 LLM 可接受的特征空间；以及一个 LLM 负责处理输入，并生成相应的语言模态的输出。在推理时，**图片会依次经过 Vision Encoder 的编码以及 Projection 的特征转换**，**并与经过 tokenizer 和 embedding 层的 instruction 进行拼接**，作为 LLM decoder layer 的输入，从而进行 LLM 的前向过程。某种意义上讲，<font style="color:#DF2A3F;">Vision Encoder 和 Projection 所扮演的角色等价于视觉的 tokenizer + embedding 层。</font>

<font style="color:rgb(25, 27, 31);">LLaVA 的模型的训练分为两个阶段：</font>

+ **<font style="color:rgb(25, 27, 31);">第一阶段，特征对齐阶段：</font>**<font style="color:rgb(25, 27, 31);">只开放 Projection 的训练，冻结 Vision Encoder 与 LLM，从而让 Projection 模块学习到特征映射关系。此时训练数据通常为图像-文本对（如图像描述或简单问答），文本通过分词器转为 token 序列，与视觉特征经投影层对齐。</font>
+ **<font style="color:rgb(25, 27, 31);">第二阶段，端到端微调过程：</font>**<font style="color:rgb(25, 27, 31);">只冻结 Vision Encoder，开放 Projection 与 LLM 的训练，从而实现多模态的对话。此阶段采用多轮对话格式数据（如用户指令、模型回复、图像上下文），文本与视觉特征拼接后输入语言模型生成连贯响应。训练数据格式统一包含图像像素和文本序列，通过特殊符号（如<image>）标记图像插入位置，实现模态融合。</font>

<font style="color:rgb(25, 27, 31);">VLM模型的训练数据格式统一包含图像像素和文本序列，通过特殊符号（如<image>）标记图像插入位置，实现模态融合。</font>

![](llava训练.png)

<font style="color:rgb(25, 27, 31);">更具体地，LLaVA的Vision Encoder 方面选用了 </font><font style="color:#DF2A3F;">CLIP-ViT-Large-Patch14（输入图片尺寸大小为 224px）</font><font style="color:rgb(25, 27, 31);">，LLM 方面选用了 Vicuna（基于 LLaMA 使用 ShareGPT 数据微调的模型），</font><font style="color:#DF2A3F;">Projection 仅为一个MLP的线性层</font><font style="color:rgb(25, 27, 31);">。随后相较于 LLaVA，LLaVA-1.5作为LLaVA的改进版，在数据和模型方面进行了进一步扩展，在性能上取得了进一步的提升。模型方面，LLaVA-1.5 将 Projection 由线性层更换为了一个两层的 MLP，将 Vision Encoder 更换为了 CLIP-ViT-Large-Patch14-336px。数据方面，LLaVA-1.5 加入了特定任务的数据集，以强化模型的表现。加入的数据集包括 VQA、OCR 以及区域感知数据。</font>


**<font style="color:#DF2A3F;">常见的VLM主要依赖如下3个核心技术：</font>**

+ **<font style="color:rgb(25, 27, 31);">视觉编码：</font>**<font style="color:rgb(25, 27, 31);">通常采用大规模预训练的视觉表征模型（如CLIP、ViT），将输入图像转换为具有语义意义的视觉特征。这一模块的核心是将像素空间映射到与语言模态对齐的离散或连续表征空间，形成视觉token序列。</font>
+ **<font style="color:rgb(25, 27, 31);">对齐机制：</font>**<font style="color:rgb(25, 27, 31);">通过可学习的投影层（如线性层或轻量级适配器），将视觉特征空间与语言模型（如LLM）的嵌入空间对齐。典型方法包括两阶段训练策略：先在大规模图文对数据上学习粗粒度对齐，再通过指令微调实现细粒度的语义融合。</font>
+ **<font style="color:rgb(25, 27, 31);">生成：</font>**<font style="color:rgb(25, 27, 31);">基于预训练的语言模型（如GPT、LLaMA），通过指令微调（Instruction Tuning）或多模态思维链（Chain-of-Thought）技术，使模型具备视觉-语言联合推理能力，最终实现从视觉输入到自然语言输出的端到端生成。</font>





<font style="color:rgb(77, 77, 77);">VLMs的分类：根据VLM的输入处理和输出生成能力将其分为三个不同的组：</font>

+ **<font style="color:rgb(51, 51, 51);">视觉语言理解模型：</font>**<font style="color:rgb(51, 51, 51);">专门为</font><font style="color:#DF2A3F;">视觉信息与语言的解释和理解而设计</font><font style="color:rgb(51, 51, 51);">的模型</font>
+ **<font style="color:rgb(51, 51, 51);">多模态输入文本生成模型：</font>**<font style="color:rgb(51, 51, 51);">擅长利用多模态输入（如图像、视频和文本）来生成文本内容</font>
+ **<font style="color:rgb(51, 51, 51);">多模态输入多模态输出模型：</font>**<font style="color:rgb(51, 51, 51);">不仅接受多模态输入，还能产生多模态的输出</font>


![](vlm分类.png)


<h2 id="ZWJYt">视觉语言理解</h2>
（Vision-Language Understanding, VLU）的VLMs专注于对**视觉信息与语言的解释和理解的结合**。

+ <font style="color:rgb(51, 51, 51);">它们设计用来处理</font><font style="color:#DF2A3F;">涉及图像和文本的复杂查询</font><font style="color:rgb(51, 51, 51);">，例如视觉问答（VQA）和图像字幕生成。</font>
+ <font style="color:rgb(51, 51, 51);">VLU模型通常需要对图像内容有深入的理解，并且能够准确地用语言来描述或回答有关图像的问题。</font>



<h3 id="ezRp9">CLIP</h3>

以CLIP进行举例说明：
论文：《Learning Transferable Visual Models From Natural Language Supervision》  
CLIP是一种神经网络，它通过自然语言指导来理解视觉概念。它能够识别多种基准上的视觉上的类别，展现出"<font style="color:#DF2A3F;">零样本"（zero-shot）</font>能力，即在没有看过特定类别样本的情况下也能识别它们。

通过对比学习的方式进行预训练，它将图像与其对应的文本描述进行对齐，从而学习视觉和语言之间的关联。

<font style="background-color:#FBDE28;">优势</font>：对于分类任务，CLIP比一般的微调深度学习视觉模型具有更强的鲁棒性。

<font style="background-color:#FBDE28;">挑战</font>：在抽象任务、细粒度分类、泛化和措辞敏感性方面仍存在困难。


<font style="color:#DF2A3F;">CLIP是通过Contrastive Learning的方式来学习Vision和文本的表征。</font>如图所示，对于<font style="color:rgb(25, 27, 31);">一个Batch的数据，以样本集中原始图文pair</font>$ <I_i,T_i> $<font style="color:rgb(25, 27, 31);">为正例pair，Batch内与其他样本的</font>$ I_x,T_x $<font style="color:rgb(25, 27, 31);">组成为负例pair：</font>$ <I_i,T_x>,<I_x,T_i>
 $<font style="color:rgb(25, 27, 31);"> 。</font>

<font style="color:rgb(25, 27, 31);">模型训练采用了对比损失函数，通过最大化正例Pair的相似度，同时最小化负例Pair的相似度来训练模型。通过这种方式，能学习到视觉特征和文本特征的对齐关系。</font><font style="color:#DF2A3F;">最后将训练好的Image Encoder模型(即ViT)参数保存下来，以供其他下游任务热启使用。</font>

![](clip.png)

CLIP的训练使用了大量成对的图像和文本数据。这些数据对通常包括一个图像及其相关的描述或标题。<font style="color:#DF2A3F;">图像编码器将输入的图像转换成一个固定大小的特征向量</font>，而<font style="color:#DF2A3F;">文本编码器将输入的文本描述转换成另一个固定大小的特征向量</font>。CLIP的核心是对图像和文本特征向量进行对比学习。**模型试图将与图像内容相匹配的文本描述的特征向量拉近，同时将不匹配的文本描述的特征向量推远。 **



<h2 id="LNekT">多模态输入的文本生成</h2>


<h3 id="RiiYi">Flamingo</h3>
<font style="color:rgb(77, 77, 77);">论文：《Flamingo: a Visual Language Model for Few-Shot Learning》</font>

<font style="color:rgb(77, 77, 77);">传统的视觉模型通常需要大量的标注数据来进行特定任务的训练和微调。然而，</font>**<font style="color:rgb(77, 77, 77);">获取大量标注数据成本高昂且耗时</font>**<font style="color:rgb(77, 77, 77);">。Flamingo 旨在通过</font>**<font style="color:rgb(77, 77, 77);">少样本学习（few-shot learning）来克服这一限制</font>**<font style="color:rgb(77, 77, 77);">，即使在只有少量示例的情况下也能快速适应新的视觉和语言任务。</font>

从整体架构来看，火鸟有几个组件：

+ Visual Encoder 模块；
+ Perceiver Resampler 模块；
+ 通过门控注意力机制 GATED XATTN-DENSE 实现的 Adapter 模块；
+ 语言模型 ChinChilla；



**<font style="color:#DF2A3F;">模型整体运行流程（将视觉皈依到语言中）</font>**：首先通过Visual Encoder 来采集视觉信息，这里视觉信息包括 图像和视频，然后使用 Perceiver Resampler 将视觉信息进行汇总，形成统一输出。紧接着，通过 Adapter模块（GATED XATTN-DENSE）将视觉信息 “嵌入” 到语言模型中，这样，就可以利用语言模型的优势来做多模态任务了。



![](fli框架.png)



+ **Visual Encoder模块（类似clip的编码器）**：使用的是 NormalizerFree ResNet (NFNet)，这个模块使用和 CLIP 一样的训练方法进行训练。之后，在火烈鸟模型中，该模块的权重会被冻结，只起到一个视觉特征提取功能，注意，上文中提到，将 CLIP 的 Visual Encoder 用在其他任务中作为特征提取模块，效果非常好，相信同样采用 CLIP 训练方法训练的 NFNet 也能具有同样的效果。



+ Perceiver Resampler模块：这个模块的作用是将图片以及不同尺寸的视频，进行统一的表征建模，从而确保其具有统一维度的输出，这样最大的好处就是模型既可以支持图片和视频，也可以保证在训练的时候，可以进行批量训练。



**<font style="color:#DF2A3F;">在视觉表示上调节冻结的语言模型，将视觉信息注入预训练的语言模型。</font>**

+ 语言部分使用的是DeepMind之前提出的具有70B参数的自回归语言模型ChinChilla。该模型和GPT系列一样，使用的是Transformer 的Decoder作为基础框架来训练的。这个模型在训练过程中是冻结的，即不更新其权重。这样做是为了避免在训练新任务时发生灾难性遗忘（catastrophic forgetting），即模型忘记了之前学到的知识。
+ 在冻结的语言模型层之间，插入了**新的交叉注意力层**，这些层是从头开始训练的。这些层被称为 <font style="color:#DF2A3F;">GATED XATTN-DENSE</font> 层，它们能够将视觉信息整合到文本生成过程中。


![](fli注意力结构.png)

+ <font style="color:rgb(51, 51, 51);">这些层由两部分组成：交叉注意力（Cross-Attention）和密集前馈网络（Dense Feed-Forward Network）。交叉注意力层使用来自感知重采样模块的视觉特征作为键（keys）和值（values），而语言输入作为查询（queries）。</font>
+ <font style="color:rgb(51, 51, 51);">GATED XATTN-DENSE 层使用了一个特殊的门控机制，通过 tanh 激活函数对新层的输出进行缩放。这个门控参数在初始化时设置为0，这意味着在训练开始时，新层的输出不会影响模型的输出，保持了冻结语言模型的完整性。</font>

<font style="color:#DF2A3F;">交叉注意力层使得模型能够在生成文本时考虑视觉输入，例如图像或视频内容。这允许模型生成与视觉场景相关的描述或回答视觉相关问题。</font>

在上文中提到，这样做的好处是，在不破坏预训练语言模型内部“知识”的情况下，可以无缝的将视觉信息“嵌入”进来，从而可以有效的利用语言模型强大的推理能力，帮助实现多模态推理。



多视觉输入支持：<font style="color:rgb(77, 77, 77);">每张图像/视频注意力掩蔽</font>

+ <font style="color:rgb(77, 77, 77);">为了有效处理多个视觉输入，Flamingo 使用了一种注意力掩蔽技术。这种技术通过限制模型在每个文本标记上可以看到的视觉标记的数量，来控制模型的注意力。</font>
+ <font style="color:rgb(77, 77, 77);">在给定的文本标记上，模型只关注在该文本标记之前出现的最后一张图像的视觉标记。这种设计允许模型专注于与当前文本标记最相关的视觉信息。</font>
+ <font style="color:rgb(77, 77, 77);">尽管模型在任何给定的文本标记上只直接关注一张图像，但通过语言模型中的自注意力机制，模型仍然可以隐式地依赖之前的所有图像。</font>


火烈鸟总结：

1，将视觉 “嵌入” 到语言模型中，这样就可以利用语言模型的知识来进行推理。

2，通过 Pereciver Resampler模块，为图片和不同时长的视频提供了一个统一的表征维度。

3，使用Adapter，在训练时，只调整Adapter，保证在不破坏语言模型所存储的知识的同时，利用语言模型进行推理，同时，由于无需调整语言模型参数，该方法非常节省计算资源。

4，通过交叉输入图片（视频）和文本的方式，训练模型，使其具有 few-shot 的多模态序列推理能力。  



<h2 id="j8wTj">总结</h2>


参考：

从Flamingo看多模态研究发展趋势[: //mp.weixin.qq.com/s/okLlk713ZR3QHtuq3-Nvag](https://mp.weixin.qq.com/s/okLlk713ZR3QHtuq3-Nvag)



当前大模型比较流行的两大趋势：

1，从<font style="color:#DF2A3F;">预训练—>微调（模型适应下游任务）</font>的思维，逐渐转变到 <font style="color:#DF2A3F;">预训练—> Prompt&amp;Adapter（下游任务适应模型）</font>的思维，而这也是当前领域正在发生的第四范式。很自然，这一范式也被逐渐引入到了多模态领域，这也是当前多模态领域发展的一个大趋势。

2，预训练语言模型本身就是一个巨型知识库，因此，在各类视觉语言任务中，通过<font style="color:#DF2A3F;">将视觉信息“嵌入”到语言模型中</font>，利用<font style="color:#DF2A3F;background-color:#FBDE28;">语言模型里面的知识来做推理</font>则是另外一个大趋势。  





<h1 id="JAvwH">2 开源视觉语言模型梳理</h1>
参考：[https://zhuanlan.zhihu.com/p/33139808097]


+ **<font style="color:rgb(25, 27, 31);">主流的VLM基座大模型：</font>**[<font style="color:rgb(9, 64, 142);">Qwen2.5-VL</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=Qwen2.5-VL&zhida_source=entity)<font style="color:rgb(25, 27, 31);">、QVQ-72B-Preview、</font>[<font style="color:rgb(9, 64, 142);">Llama3.2-Vision</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=Llama3.2-Vision&zhida_source=entity)<font style="color:rgb(25, 27, 31);">、</font>[<font style="color:rgb(9, 64, 142);">InternVL2.5-MPO</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=InternVL2.5-MPO&zhida_source=entity)<font style="color:rgb(25, 27, 31);">主要是在服务端运行的超大VLM模型。</font>
+ **<font style="color:rgb(25, 27, 31);">中小尺寸的轻量化多模态模型：</font>**[<font style="color:rgb(9, 64, 142);">MiniCPM-o</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=MiniCPM-o&zhida_source=entity)<font style="color:rgb(25, 27, 31);">、</font>[<font style="color:rgb(9, 64, 142);">Phi-4-multimodal</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=Phi-4-multimodal&zhida_source=entity)<font style="color:rgb(25, 27, 31);">、</font>[<font style="color:rgb(9, 64, 142);">Gemma3</font>](https://zhida.zhihu.com/search?content_id=255650449&content_type=Article&match_order=1&q=Gemma3&zhida_source=entity)<font style="color:rgb(25, 27, 31);">主要是中小尺寸的多模态模型，其模型设计初衷只希望在算力有限的设备上运行，同时部分模型还支持音频输入与生成。</font>



![](开源vlm.png)


<h2 id="XCCyf">Qwen视觉语言模型梳理</h2>
参考：[https://zhuanlan.zhihu.com/p/25267823390]


Qwen系列模型是由阿里巴巴开源的基座系列模型。是目前全球受众最广泛，影响力最大的基座模型之一。Qwen关于VLM模型发布分别为：Qwen-VL，Qwen2-VL，Qwen2.5-VL，共有8个不同尺寸的模型。

<font style="color:rgb(25, 27, 31);">QWenVL出来的时间相对较晚，所以最开始就采用了以LLM为核心、将图像特征简单转换为文本特征的框架。其中，</font><font style="color:#DF2A3F;">QWenVL是第一个版本，训练的模型较小（～9.6B）</font><font style="color:rgb(25, 27, 31);">，并且不支持高清分辨率的图像；</font><font style="color:#DF2A3F;">QWen2-VL将模型的参数量增加到～70B</font><font style="color:rgb(25, 27, 31);">，支持动态的分辨率；</font><font style="color:#DF2A3F;">QWen2.5-VL将模型的数据量增加到4.1T token</font><font style="color:rgb(25, 27, 31);">，并且借鉴了LLM中的RL和</font>[<font style="color:rgb(9, 64, 142);">COT</font>](https://zhida.zhihu.com/search?content_id=256265365&content_type=Article&match_order=1&q=COT&zhida_source=entity)<font style="color:rgb(25, 27, 31);">技术。</font>

下面是几个版本的对比：

| **模型** | **<font style="color:rgb(25, 27, 31);">QWen-VL</font>** | **<font style="color:rgb(25, 27, 31);">QWen2-VL</font>** | **<font style="color:rgb(25, 27, 31);">QWen2.5-VL</font>** |
| --- | --- | --- | --- |
| 发布时间 | <font style="color:rgb(25, 27, 31);">2023.8</font> | <font style="color:rgb(25, 27, 31);">2024.9</font> | <font style="color:rgb(25, 27, 31);">2025.2</font> |
| 模型结构 | <font style="color:rgb(25, 27, 31);">Qwen-7B+</font><font style="color:rgb(9, 64, 142);">ViT</font><font style="color:rgb(25, 27, 31);">-bigG-1.9B+Adapter</font> | <font style="color:rgb(25, 27, 31);">Vision Encoder：675M   </font><font style="color:rgb(25, 27, 31);">LLM：Qwen2:1.5B~72B   </font><font style="color:rgb(25, 27, 31);">Vision-Language Merger</font> | <font style="color:rgb(25, 27, 31);">Vision Encoder: relatively fewer parameters   </font><font style="color:rgb(25, 27, 31);">LLM：Qwen2.5:3B~72B   </font><font style="color:rgb(25, 27, 31);">Vision-Language Merger</font> |
| Vision Encoder网络结构 | <font style="color:rgb(25, 27, 31);">OpenClip's ViT-bigG</font> | <font style="color:rgb(25, 27, 31);">DFN*vit + 2D RoPE</font> | <font style="color:rgb(25, 27, 31);">Dynamic-resolution ViT+Window Attention*+2D RoPE</font> |
| <font style="color:rgb(25, 27, 31);">Adaptor网络结构</font> | <font style="color:rgb(25, 27, 31);">Cross attention module</font> | <font style="color:rgb(25, 27, 31);">MLP</font> | <font style="color:rgb(25, 27, 31);">MLP</font> |
| <font style="color:rgb(25, 27, 31);">训练数据</font> | <font style="color:rgb(25, 27, 31);">1.4B+76.8M+ 350k</font> | <font style="color:rgb(25, 27, 31);">1.4 trillion tokens</font> | <font style="color:rgb(25, 27, 31);">4.1 trillion tokens</font> |
| <font style="color:rgb(25, 27, 31);">创新点</font> | <font style="color:rgb(25, 27, 31);">与主流的框架差别很小，工程上的贡献多于学术贡献</font> | <font style="color:#DF2A3F;">支持不同分辨率的图像输入   </font><font style="color:#DF2A3F;">MRoPE</font> | <font style="color:#DF2A3F;">VLM RLHF SFT+DPO   </font><font style="color:#DF2A3F;">MCOT</font> |
| 版本参数 | Qwen-VL：9.6B | <font style="color:rgb(25, 27, 31);">Qwen2-VL-2B：2.2B</font><br/><font style="color:rgb(25, 27, 31);">Qwen2-VL-7B：8.3B</font><br/><font style="color:rgb(25, 27, 31);">Qwen2-VL-72B：73B</font> | <font style="color:rgb(25, 27, 31);">Qwen2.5-VL-3B：3B</font><br/><font style="color:rgb(25, 27, 31);">Qwen2.5-VL-7B：8.3B</font><br/><font style="color:rgb(25, 27, 31);">Qwen2.5-VL-72B：73B</font><br/><font style="color:rgb(25, 27, 31);"></font> |
| 链接 | [https://huggingface.co/Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) | [https://huggingface.co/Qwen/Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B)<br/>[https://huggingface.co/Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B)<br/>[https://huggingface.co/Qwen/Qwen2-VL-72B](https://huggingface.co/Qwen/Qwen2-VL-72B) | [https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)<br/>[https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)<br/>[https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) |


<font style="color:rgb(83, 88, 97);">注： 上表只列出了几个主要的版本，并没有列出一些衍生版本的模型，Qwen发布的模型还包括：量化、指令微调等模型版本</font>



下面对QwenVL系列模型进行具体介绍和学习：

<h3 id="djfau">Qwen-VL</h3>
论文：《Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond》

论文链接：[https://arxiv.org/pdf/2308.12966](https://arxiv.org/pdf/2308.12966)



<font style="color:rgb(25, 27, 31);">Qwen-VL 于 2023 年 8 月发布，和LLaVA在模型结构设计和训练策略上有诸多不同：在大语言模型（LLM）方面，使用了来自 Qwen-7B 模型的预训练权重；视觉编码器采用 Vision Transformer（ViT）架构处理输入图像生成图像特征，训练和推理时会将输入图像</font><font style="color:#DF2A3F;">调整到特定分辨率并分割成大小为 14 的图块</font><font style="color:rgb(25, 27, 31);">，整个ViT的预训练权重源自 OpenAI 的 </font><font style="color:#DF2A3F;">CLIP 的 ViT-bigG</font><font style="color:rgb(25, 27, 31);">；Projector采用位置感知的视觉-语言Adaptor，</font><font style="color:#DF2A3F;">用于缩短图像 token 长度并作模态映射，由单层交叉注意力模块构成，以一组可训练向量</font><font style="color:rgb(25, 27, 31);">（嵌入）为查询向量、视觉编码器的图像特征为交叉注意力操作的键且随机初始化。</font>

<font style="color:rgb(25, 27, 31);">套用MM-LLM的框架，Qwen-VL包括3个典型的模块：</font>

+ **<font style="color:rgb(25, 27, 31);">模态编码器（Modality Encoder）</font>**<font style="color:rgb(25, 27, 31);"> </font><font style="color:rgb(25, 27, 31);">： 视觉编码器（visual encoder），只用来编码图片视觉特征</font>
+ **<font style="color:rgb(25, 27, 31);">输入投影层（Input Projector）</font>**<font style="color:rgb(25, 27, 31);">：位置感知的适配器（position-aware adapter）</font>
+ **<font style="color:rgb(25, 27, 31);">LLM主干网络（LLM Backbone）</font>**<font style="color:rgb(25, 27, 31);">： Qwen-7B Base 模型</font>

<font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">下面我们分别来看看Qwen-VL的两个核心模块：</font>**<font style="color:rgb(25, 27, 31);">视觉编码器、感知位置的适配器。</font>**<font style="color:rgb(25, 27, 31);">接着描述一些规范化样本处理过程，最后描述下模型的训练过程</font>**<font style="color:rgb(25, 27, 31);">。</font>**

<h4 id="AS0mh"><font style="color:rgb(25, 27, 31);">视觉编码器 Visual Encoder</font></h4>
<font style="color:rgb(25, 27, 31);">Qwen-VL的视觉编码器使用的是</font><font style="color:#DF2A3F;">ViT架构（Vision Transformer）</font><font style="color:rgb(25, 27, 31);">，ViT的网络设置和初始化参数使用了OpenCLIP预训练好的</font><font style="color:#DF2A3F;">ViT-bigG模型</font><font style="color:rgb(25, 27, 31);">。OpenCLIP是laion.ai组织的一个开源项目，是对OpenAI's的CLIP（Contrastive Language-image Pre-training）的开源实现。laion.ai发布了一系列基于CLIP框架训练的不同size模型。</font>

<font style="color:rgb(25, 27, 31);">Clip：huggingface模型：</font>[https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)



<font style="color:rgb(25, 27, 31);">在Qwen-VL中采用的是标准的ViT框架，ViT的原理比较简单：将图片分割成多个</font><font style="color:#DF2A3F;">图像块（Patch</font><font style="color:rgb(25, 27, 31);">），然后针对每个Patch通过一系列线性映射，</font><font style="color:#DF2A3F;">转化成token</font><font style="color:rgb(25, 27, 31);">，再将所有token拼接成序列，最终将一张图片从</font>$ (H,W,C) $<font style="color:rgb(25, 27, 31);">格式转换成</font>$ (S,H) $<font style="color:rgb(25, 27, 31);">格式的序列特征。其中  H：高，W：宽，C：通道数，S： 序列长度，H：特征维度。在标准的ViT实现上，输入图片会先被调成1:1长宽比的正方形，然后再分割成固定的图像块。</font>

<font style="color:rgb(25, 27, 31);">因此这种标准的ViT框架的设计，</font><font style="color:#DF2A3F;">只能接收固定分辨率的图片</font><font style="color:rgb(25, 27, 31);">，同时Patch的大小也是模型在训练期间使用的一个固定size。ViT处理过程如图所示：</font>


![](qwenvl1.png)

根据上面的知识我们可以再看Qwen-VL的编码器过程（源码：[https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py](https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py)）：



<font style="color:rgb(83, 88, 97);">首先看下源码设置的一些参数：</font>

<font style="color:rgb(83, 88, 97);">Qwen-VL 可接受的图像分辨率为448*448，所以输入的图片会先处理成统一的尺寸：（W: 448, H: 448，C: 3）。注：Qwen-VL模型训练其实是做了三个阶段。第一阶段图像会统一处理成低像素224*224，后面两个阶段统一分辨率为448*448。这里只以高分辨率的设置为例。</font>

<font style="color:rgb(83, 88, 97);">patch_size : 14 ，这个参数指定patch的大小，同时也是卷积核和的尺寸，也是卷积操作的stripe步长</font>

<font style="color:rgb(83, 88, 97);">width：1664，这个参数指定的输出通道数，即out_channels，也就是每个Patch输出的特征的维度</font>  
<font style="color:rgb(83, 88, 97);">我们以batch_size(B) = 1为例</font>



<font style="color:rgb(25, 27, 31);">ViT核心处理就几行代码，如下：</font>

```python
class VisionTransformer(nn.Module):
   def __init__(...）:
       self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

   def forward(self, x: torch.Tensor):
        # 注释1：通过卷积核将一张图片从[H，W，C]=[448, 448, 3] 映射成 [width, grid, grid] = [1664, 32, 32]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 注释2：一张图片按行展开，[width, grid, grid] 映射成 [grid * grid, width]二维序列
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 注释3：增加位置编码输入transformer模型
        x = x + get_abs_pos(self.positional_embedding, x.size(1))
        x = self.transformer(x)
```



**<font style="color:rgb(25, 27, 31);">代码注释1的处理过程</font>**<font style="color:rgb(25, 27, 31);">：一张图片做卷积操作，处理成 [width, grid, grid] = [1664, 32, 32]的数据，如图所示：</font>


![](qwen-vit1.png)

**<font style="color:rgb(25, 27, 31);">代码注释2的处理过程</font>**<font style="color:rgb(25, 27, 31);">：按行优先展开，处理成一个二维格式的数据[sequence_len, hidden_size] = [1024, 1664]（类似与一条文本处理后的序列）。如图下所示。</font>

![](qwen-vit2.png)

<h4 id="HqXDY">输入投影层：位置感知的适配器 Position-aware Vision-Language Adapter</h4>
<font style="color:rgb(25, 27, 31);">经过上述ViT处理后，对于448*448分辨率的图像，生成一个[1024, 1664]的序列，也就是向量维度为1664的长度为1024的序列。为了压缩视觉token的输入长度，Qwen-VL引入了一个Adapter来压缩图像特征。这个Adaper就是一个随机初始化的单层</font>**<font style="color:rgb(25, 27, 31);">Cross-Attention模块</font>**<font style="color:rgb(25, 27, 31);">。该模块使用一组可学习的query向量，将来自ViT的图像特征作为Key向量。通过Cross-Attention操作后将视觉特征序列压缩到</font>**<font style="color:rgb(25, 27, 31);">固定的256长度</font>**<font style="color:rgb(25, 27, 31);">。</font>

> <font style="color:rgb(83, 88, 97);">对于Transformer我们平时接触更多的是Self-Attention，在Self-Attention计算中q,k,v都是基于输入特征做矩阵变换后得到的，通常q,k,v的长度处理前后也是一样的。</font>  
<font style="color:rgb(83, 88, 97);">那么这里提到的Cross-Attention、可学习的query向量、做序列压缩等，针对这些描述是否真正理解了呢？</font>
>

如下图所示<font style="color:rgb(25, 27, 31);">描述了基于可学习Query和ViT输出的序列作为k，v的Attention计算过程，经过Cross-Attention后，将ViT阶段的1024长度的序列，压缩到了长度为256的序列。</font>

![](qwen-vit3.png)

<font style="color:rgb(25, 27, 31);">此外，考虑到位置信息对于精细图像理解的重要性，Qwen-VL将</font>**<font style="color:rgb(25, 27, 31);">二维绝对位置编码（三角位置编码）</font>**<font style="color:rgb(25, 27, 31);">整合到Cross-Attention的q,k中，以减少压缩过程中可能丢失的位置细节。随后将长度为256的压缩图像特征序列输入到大型语言模型中。</font>

<h4 id="JzBPJ">输入和输出</h4>
对于输入LLM前的特征序列，为了区分图片和文本的输入信息，对图片的feature使用了特殊的token包裹，图像特征的开始和结束用<img>和</img>token 圈定，来明确标识图像特征的起止位置。<font style="color:rgb(25, 27, 31);">同时为了做grounding任务，对图像中bounding box 统一用一个"左上-右下"坐标框格式表示："</font>

$ (X_topleft,Y_topleft),(X_bottomright,Y_bottomright) $。<font style="color:rgb(25, 27, 31);">坐标值统一做归一化处理，规范化到（0，1000）区间。并用<ref>,</ref>两个特殊的token圈定起来。下面图7是一条典型的grounding 任务的样本实例：</font>


![](qwen4.png)

![](qwen5.png)



<h4 id="YUbXA">训练过程</h4>


<font style="color:rgb(25, 27, 31);">Qwen-VL 的训练流程包含3个阶段，分别为预训练、多任务预训练和监督微调。</font>

![](qwen训练.png)



+ **<font style="color:rgb(25, 27, 31);">第一阶段，单任务大规模预训练</font>**<font style="color:rgb(25, 27, 31);">：冻结大语言模型，仅优化视觉编码器和基于Cross Attention构建的 VL Adaptor 。利用大规模、弱标记、网络爬取的图像文本对（清洗后 14 亿对，77.3% 为英语、22.7% 为中文），训练数据的图片统一处理成224*224的尺寸。</font>
+ **<font style="color:rgb(25, 27, 31);">第二阶段，多任务预训练</font>**<font style="color:rgb(25, 27, 31);">：解锁大语言模型，训练整个模型。引入高质量、细粒度的 VL 注释数据，在 7 个任务（如字幕、视觉问答、定位等）上同时训练，使用多个公开数据集和内部语料库，该阶段的训练数据，Vision数据的分辨率从224提升到448。</font>
+ **<font style="color:rgb(25, 27, 31);">第三阶段，SFT有监督微调</font>**<font style="color:rgb(25, 27, 31);">：冻结视觉编码器，优化语言模型和适配器模块。多模态指令调整数据来自字幕或对话数据，还构建额外对话数据，混合多模态和纯文本对话数据（35 万条）。通过指令微调增强指令跟随和对话能力，得到 Qwen-VL-Chat 模型。</font>







<h3 id="hNrOO">Qwen2-VL</h3>
链接：

<font style="color:rgb(25, 27, 31);">相对于Qwen-VL，Qwen2-VL整体模型架构做的比较大的升级，首先从模型命名上可知，主体模型从Qwen升级到了Qwen2。并且发布了三个size的模型，分别是Qwen2-VL-2B，Qwen2-VL-7B，Qwen2-VL-72B。</font>

> 注：
>
> <font style="color:rgb(83, 88, 97);">1. Qwen2-VL系列模型，针对Vision Encoder采用了相同size的模型结构，这里应该是做了一些ablation的实验，取得一个合适的size。</font>  
<font style="color:rgb(83, 88, 97);">2. 另外相对于Qwen-VL系列，Qwen2-VL并没有显示描述Vision-Language Adapter的参数，通过查看源码，Qwen2-VL对Adapter做了简化处理，并没有采用一个Cross-Attention的结构，而是使用了简单的线性变换层，这层参数比较少，相对于总参数规模，可以忽略不计。</font>
>



主要有以下重要的升级点：

+ 1，**<font style="color:rgb(25, 27, 31);">采用原生动态分辨率：单一分辨率 -> 任意分辨率，</font>**<font style="color:rgb(25, 27, 31);"> Qwen-VL模型输入只接受单一分辨率的图片，Qwen2-VL 引入了</font><font style="color:#DF2A3F;">朴素动态分辨率机制，能处理任意分辨率的图像</font><font style="color:rgb(25, 27, 31);">，将其动态转换为可变数量的视觉令牌。通过修改了 ViT结构，去除原始绝对位置嵌入，引入 2D-RoPE 来捕获图像二维位置信息。在推理阶段，不同分辨率图像被打包成单个序列，还通过 MLP 层压缩视觉token，提高了处理效率和对不同分辨率图像的适应性。</font>
+ <font style="color:rgb(25, 27, 31);">2，</font>**<font style="color:rgb(25, 27, 31);">Vision Encoder位置编码：绝对位置编码 -> 相对位置编码</font>**<font style="color:rgb(25, 27, 31);">，从二维三角位置编码升级到二维RoPE位置编码，RoPE对长序列有更好的泛化能力，有利于提升对长序列Vision特征的建模能力。</font>
+ <font style="color:rgb(25, 27, 31);">3，</font>**<font style="color:rgb(25, 27, 31);">LLM主体模型位置编码</font>**<font style="color:rgb(25, 27, 31);">：</font>**<font style="color:rgb(25, 27, 31);">1D->3D RoPE</font>**<font style="color:rgb(25, 27, 31);">，引入</font><font style="color:#DF2A3F;">多模态旋转位置编码技术（M-RoPE）</font><font style="color:rgb(25, 27, 31);">，刻画多模态(时序、高、宽)三维数据。进一步提升对时空数据的建模能力。</font>
+ <font style="color:rgb(25, 27, 31);">4，</font>**<font style="color:rgb(25, 27, 31);">统一多模态数据： 单图片 -> 统一图片和视频，</font>**<font style="color:rgb(25, 27, 31);">统一框架处理图片和视频数据，进一步提升对真实世界认知和理解能力。</font>
+ <font style="color:rgb(25, 27, 31);">5，</font>**<font style="color:rgb(25, 27, 31);">训练数据： 1.4B -> 1.4T</font>**<font style="color:rgb(25, 27, 31);">，数据量提升了3个量级，同时数据覆盖了多领域任务。</font>



下面详细介绍下这些升级点：

<h4 id="ebKuG">原生动态分辨率 Naive Dynamic Resolution</h4>
<font style="color:rgb(25, 27, 31);">Qwen-VL使用的视觉编码器是标准的ViT，这要求输入的图片要统一处理成单一的、固定的分辨率，才能feed到模型进行处理。一般标准的预训练好的ViT，通常是将图片处理成正方形（长:宽=1:1）。这样处理后通常图片会失真，导致模型理解上有信息损失或引入一些误导。如下图所示：</font>


![](原生动态.png)

<font style="color:rgb(25, 27, 31);">左侧是传统的ViT对输入的处理（也是Qwen-VL采用的方法），对于一些宽高比差距较大的图片，处理后通常会造成图片扭曲，而Qwen2-VL实现的</font>**<font style="color:rgb(25, 27, 31);">原生动态分辨率方法</font>**<font style="color:rgb(25, 27, 31);">则会保留原始图片的宽高比，将图片resize到适当的大小，图片像素满足[min_pixel, max_pixel]区间,再对图片做Patch处理，将每个图片处理成变长的Vision token序列，再输入给LLM模型。</font>

<font style="color:rgb(25, 27, 31);">目前看上述的方法是比标准的ViT更合理的，因为它保留了图片的原始分辨率，但是同时也引入了一个问题。</font>

> <font style="color:rgb(83, 88, 97);">问题是这样：</font>  
<font style="color:rgb(83, 88, 97);">传统的ViT会将任何图片数据都处理成定长的Patch序列，然后输入给Vision Encoder，这种统一定长的输入是对硬件计算非常友好的，非常好组Batch，并且不需要任何padding处理。Batch序列中每个位置的计算都是有效的。</font>  
<font style="color:rgb(83, 88, 97);">而对于上面提到的原生动态分辨率方法会将不同图片处理成不同长度的Patch序列。对于不同的长度的输入，做并行计算时，我们自然会想到类似于文本数据的操作，对数据做padding，再Feed给模型。但这相比传统的ViT方法（无Padding）会更慢（因为为了适配一个Batch中最长的序列，要做适当的Padding处理，导致会有些冗余计算）。因此这并不是一个完美的方法。Qwen2-VL采用的原生动态分辨率方法实现上同时也考虑了性能问题。</font>
>



<font style="color:rgb(25, 27, 31);">那么原生动态分辨率方法具体是怎么实现的呢？ </font>**<font style="color:rgb(25, 27, 31);">核心方法是采用了NaViT的Patch Pack技术，把不同图像的多个patch打包到一个序列，能保留不同图片的可变分辨率。同时在一个次序列计算中同时可处理多个图像，提升了模型计算的吞吐，在性能上始终优于传统的ViT</font>**<font style="color:rgb(25, 27, 31);">。其性能提升主要来源于Pack处理后，一个序列包括多个图片能同时计算，使得在固定计算预算下，动态分辨率方法能训练更多样本，从而带来更好的性能。</font>

<font style="color:rgb(25, 27, 31);">那么一个序列中塞进了多个图像数据，</font><font style="color:#DF2A3F;">怎么能互不干扰的计算呢</font><font style="color:rgb(25, 27, 31);">（也就是在做ViT的Attention计算时，多个图片的Patch在一个序列中需要做计算隔离）</font>

<font style="color:rgb(25, 27, 31);">我们以一个简单例子描述下动态分辨率方法的处理逻辑。</font>

> **<font style="color:rgb(83, 88, 97);">举例</font>**<font style="color:rgb(83, 88, 97);">：假设我们5张图片:</font>$ I_1-I_5 $<font style="color:rgb(83, 88, 97);">，且patch长度为2~6，即图片Patch后长度为： </font>$ {I_1:2,I_2:3,I_3:4,I_4:5,I_5:6}
 $<font style="color:rgb(83, 88, 97);"> 。为了描述简单，我们假设模型设置Batch_Size=2，并且正好处理这5张图片到一个Batch中。</font>
>

<font style="color:rgb(25, 27, 31);">处理过程：</font>

**<font style="color:rgb(25, 27, 31);">a）首先我们将5张图片进行Pack，放到2个序列中</font>**

<font style="color:rgb(25, 27, 31);">一个很简单的方式是将3个Patch较短的图片放到一个序列</font>$ S_1 $<font style="color:rgb(25, 27, 31);">，2个较长Patch的图片放到一个序列</font>$ S_2 $<font style="color:rgb(25, 27, 31);">。符号化为：Batch = </font>$ ({S_1,S_2}) $<font style="color:rgb(25, 27, 31);">  ，其中 </font>$ S_1=(I_1:2,I_2:3,I_3:4) $<font style="color:rgb(25, 27, 31);"> 序列长度为 9，</font>$ S_2 = (I_4:5,I_5:6) $<font style="color:rgb(25, 27, 31);">序列长度为11。</font>

<font style="color:rgb(25, 27, 31);">b)  </font>**<font style="color:rgb(25, 27, 31);">Batch内做序列Padding对齐处理</font>**

<font style="color:rgb(25, 27, 31);">根据Batch内最长序列，通过F.pad方法做序列对齐，在序列前后增加Padding token，该例子中由于  较短，需要在末尾增加Padding token，处理后，如下图所示  
</font>
![](batch pack.png)



**<font style="color:rgb(25, 27, 31);">c) 通过设置Attention Mask保证同Sequence中各图片计算隔离</font>**

<font style="color:rgb(25, 27, 31);">一个序列中有多张图片输入，在计算时要必须保证各图片的Attention计算是相互隔离的。实现上通过对Attention Mask矩阵做特殊的设置，来保证计算隔离。计算Attention Mask的过程如下：</font>

<font style="color:rgb(25, 27, 31);">首先，记录序列中每个图片起止token位置（包括初始0位置），得到两个位置序列为</font>$ P_{s1}={0,2,5,9} $<font style="color:rgb(25, 27, 31);">和 </font>$ P_{s2}={0,5,11} $<font style="color:rgb(25, 27, 31);">.</font>

<font style="color:rgb(25, 27, 31);">然后，分别用</font>$ P_{s1} $<font style="color:rgb(25, 27, 31);">和</font>$ P_{s2} $<font style="color:rgb(25, 27, 31);">来计算二维Attention mask矩阵，计算方式为：先初始化一个全0的mask矩阵，然后遍历每个</font>$ P_{st} $<font style="color:rgb(25, 27, 31);">,取[i,i+1]位置的两个数字（j,k）,使得矩阵行列坐标都满足在[j,k-1] 区间范围的位置置1。两个序列计算后的Mask矩阵，如下图所示。</font>

![](pack.png)



<font style="color:rgb(25, 27, 31);">计算好了上面的Attention Mask矩阵，在过Vision Encoder网络时，将Attention Mask作用在Attention计算上，就会隔离同一序列中不同图像的Attention计算。</font>



<h4 id="j9282">2D-RoPE位置编码 <font style="color:#DF2A3F;"> </font></h4>
<font style="color:rgb(25, 27, 31);">在Qwen2-VL系列的ViT网络中，并没有沿用Qwen-VL的2D绝对位置编码，而是引入了2D-RoPE相对位置编码。之所以引入2D-RoPE，我个人理解主要考虑Qwen2-VL系列处理的图片Patch是变长的，对于超长的一些位置，如果采用绝对位置编码，由于数据稀疏性， 并不能得到充分训练。但RoPE本身是具有一定的外推性，对长序列建模有更好的泛化能力。</font>

<font style="color:rgb(25, 27, 31);">1维的旋转位置编码（1D-RoPE）对序列增加相对位置的处理过程。这里简单引用苏神的推导结论：</font>[https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)



首先对序列的每个位置构建分块矩阵，形如：

![](rope.png)

<font style="color:rgb(25, 27, 31);">其中 m表示序列的位置， </font>$ \theta_i 
 $<font style="color:rgb(25, 27, 31);"> 沿用Sinusoidal位置编码的取值, d为位置编码向量的维度。</font>

<font style="color:rgb(25, 27, 31);">在计算Attention时，计算q,k乘积前，要首先对 q,k 做变换，也就是给  m位置的 q 乘矩阵  Rm，给  n位置的k乘以矩阵 Rn 。这样计算的 q,k通过增加绝对位置的变换，实质上是增加了相对位置信息。如下公式：</font>

![](rope2.png)

<font style="color:rgb(25, 27, 31);">由于上述 Rm 变换矩阵比较稀疏，直接用矩阵乘法来实现会浪费算力，苏神也给出了一个推荐的实现方式，如下：</font>

![](rope3.png)

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">现在我们知道1维旋转位置编码RoPE的计算方式，那么怎么扩展到2维呢？参考苏神另一篇博客（详见：</font>[二维位置的旋转式位置编码](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/8397)<font style="color:rgb(25, 27, 31);">）。</font>**<font style="color:rgb(25, 27, 31);">RoPE从1维扩展到2维一个简单的结论：针对一个位置</font>**<font style="color:rgb(25, 27, 31);"> （x,y） ，</font>**<font style="color:rgb(25, 27, 31);">对维度为 d 的输入向量分成两半，前一半向量用 x 的一维RoPE矩阵( Rx )处理，后一半向量用 y 的一维RoPE矩阵( Ry )处理，然后再将两半处理后的结果拼接在一起，就做完了2维的RoPE处理。</font>**<font style="color:rgb(25, 27, 31);">（相对与一维RoPE，扩展到二维，操作是比较简单，具体原理上请参考</font>[苏神的博客)](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/8397)

<font style="color:rgb(25, 27, 31);"></font>

<h4 id="edHNu"><font style="color:rgb(25, 27, 31);">输入投影层：压缩Vision token + MLP Adapter</font></h4>
<font style="color:rgb(25, 27, 31);">Qwen-VL在输入投影层做了Vision token的压缩处理，是采用了Cross-Attention的架构，通过一个组可学习的Query向量来压缩原始的特征序列。那么Qwen2-VL为什么没有继续沿用Cross-Attention的架构？</font>

<font style="color:rgb(25, 27, 31);">这里主要是因为Cross-Attention架构适合处理固定长度的 k,v ，</font><font style="color:#DF2A3F;">当 k,v 长短不一时，是不适合做Attention计算的。</font><font style="color:rgb(25, 27, 31);">而Qwen2-VL通过原生动态分辨率方法处理的每个图片的token序列恰恰是变长的，无法使用Cross-Attention架构做特征压缩处理。</font>

<font style="color:rgb(25, 27, 31);">Qwen2-VL采用了一种更简单的压缩方法：</font><font style="color:#DF2A3F;">对空间位置临近的patch 特征做拼接，再经过2层MLP线性变换，这样将原来长度为 n 的序列，可压缩到 n/4 </font><font style="color:rgb(25, 27, 31);">，最终将压缩后的特征序列输入给LLM模型。处理过程如图所示：</font>

![](project.png)

<font style="color:rgb(25, 27, 31);">为了区分Vision token和文本token，Qwen2-VL也引入了两个特殊的token，<|vision_start|>和<|vision_end|>来标识Vision token。</font>

> <font style="color:rgb(83, 88, 97);">对于一个 224*224，如果ViT的 patch_size = 14，最终将图片编码成一个66个token的序列输入到模型。  
</font><font style="color:rgb(83, 88, 97);">具体计算过程：  
</font><font style="color:rgb(83, 88, 97);">1.Patch 处理后的Token数为： (224/14)*(224/14) = 16*16 = 256  
</font><font style="color:rgb(83, 88, 97);">2.经过输入投影层压缩处理： 256/4 = 64  
</font><font style="color:rgb(83, 88, 97);">3. 最后再加上  个起止位置的特殊token：64+2 = 66 </font>
>

<font style="color:rgb(25, 27, 31);">  
</font>

<h4 id="VkSPN">Multimodel Rotary Position Embedding(M-RoPE)</h4>
<font style="color:rgb(25, 27, 31);">Qwen2-VL模型输入增加了视频模态，视频可以看做是在图片二维空间上，增加了时序维度，是三维时空分布的数据: (T,H,W)，M-RoPE将位置编码信息从1维扩展到了3维，这样就能清晰刻画视频模态数据时空位置信息。对于文本（一维）和图像（二维）的数据如何统一表示成3维的位置ID呢？处理也比较简单直接：</font>

+ <font style="color:rgb(25, 27, 31);">文本：因为文本是一维空间序列，三个维度的值保持一致，也就退化成1D-RoPE。</font>
+ <font style="color:rgb(25, 27, 31);">图像：图像只有宽高两个维度，所以对于一张图片，时序维度  的位置始终保持固定。</font>
+ <font style="color:rgb(25, 27, 31);">对于混合多模态数据，每个模态的起始position ID是前面模态三维位置ID中取最大的ID并加1得到。</font>

<font style="color:rgb(25, 27, 31);">有了三维的位置，最终怎么映射成3D-RoPE，映射方式类似与2D-RoPE，</font>**<font style="color:rgb(25, 27, 31);">针对一个位置（x,y,z)</font>**<font style="color:rgb(25, 27, 31);">，</font>**<font style="color:rgb(25, 27, 31);">对维度为d的输入向量分成三份，前一份向量用x的一维RoPE矩阵( Rx )处理，中间一份向量用 y 的一维RoPE矩阵( Ry )处理，最后一份向量用</font>**<font style="color:rgb(25, 27, 31);"> z</font>**<font style="color:rgb(25, 27, 31);"> 的一维RoPE矩阵（</font>**<font style="color:rgb(25, 27, 31);"> Rz </font>**<font style="color:rgb(25, 27, 31);">）处理，然后再将三份处理后的结果拼接在一起，就做完了3维的RoPE处理。</font>**

![](m-rope.png)



<h4 id="EtZGg">统一的图像和视频理解框架；</h4>
<font style="color:rgb(25, 27, 31);">Qwen2-VL统一了视频和图像的理解框架，能混合输入图像和视频数据进行理解。为了保证图片和视频的处理一致，对视频和图像分别做如下处理：</font>

<font style="color:rgb(25, 27, 31);">视频处理：以每秒两帧的速率对视频进行采样，最终可采样偶数个帧序列。对于长视频为了平衡序列长度和计算效率，通过动态调整每一帧的分辨率，将视频总token限制在16K以内。</font>

<font style="color:rgb(25, 27, 31);">图像处理：对图像做复制操作，使得单一图片，变成一个时序为2的帧序列。</font>

<font style="color:rgb(25, 27, 31);">使用3D的卷积对帧序列做特征抽取，如图所示，每两张图片为一组进行卷积操作抽取特征。这样通过将卷积核扩充了时序维度，可以进一步压缩序列长度，因此也能进一步提升模型处理更多帧的能力。</font>

![](统一.png)



<h4 id="hjyGR">模型训练</h4>
<font style="color:rgb(25, 27, 31);">Qwen2-VL采用了与Qwen-VL一致的三阶段训练方式，但是Qwen2-VL在训练数据上相比Qwen-VL做了大量的有价值的工作。</font>

<font style="color:rgb(25, 27, 31);">数据来源除了获取开源数据、经过清洗的网页数据，还做的大量数据合成的工作。数据涉及多种场景，包括图像-文本对，OCR数据，视觉问答数据，视频对话数据等多样化数据。</font>

<font style="color:rgb(25, 27, 31);">此外Qwen2-VL数据规模大幅提升，Qwen-VL整体训练样本1.4B左右，Qwen2-VL直接翻了3个量级达到了1.4T。</font>

<font style="color:rgb(25, 27, 31);">通过大幅提升样本规模和样本多样性，使得Qwen2-VL的模型效果在多任务的评估中保持领先，也碾压了GPT-4o的效果。</font>

![](模型训练.png)




<h3 id="g6ZRp">Qwen2.5 -VL</h3>
<font style="color:rgb(25, 27, 31);">Qwen2.5-VL可循的材料只有一篇官方的</font>[博客](https://link.zhihu.com/?target=https%3A//qwenlm.github.io/zh/blog/qwen2.5-vl/)<font style="color:rgb(25, 27, 31);">，官方的一张图基本描述了相对于Qwen2-VL的一些更新。Qwen2-VL 相比，Qwen2.5-VL 增强了模型对时间和空间尺度的感知能力，并进一步简化了网络结构以提高模型效率。</font>

![](qwen2.5.png)



升级点：

+ 时间和图像尺寸的感知

在空间维度上，Qwen2.5-VL 不仅能够动态地将不同尺寸的图像转换为不同长度的 token，**<font style="color:#DF2A3F;">还直接使用图像的实际尺寸来表示检测框和点等坐标</font>**，而不进行传统的坐标归一化。这使得模型能够直接学习图像的尺度。在时间维度上，引入了动态 FPS (每秒帧数)训练和绝对时间编码，<font style="color:#DF2A3F;">将 mRoPE id 直接与时间流速对齐。这使得模型能够通过时间维度 id 的间隔来学习时间的节奏。</font>



> <font style="color:rgb(108, 108, 108);background-color:rgb(245, 245, 245);">User</font>
>
> <font style="color:rgb(31, 31, 31);background-color:rgb(245, 245, 245);">Detect all motorcyclists in the image and return their locations in the form of coordinates. The format of output should be like {“bbox_2d”: [x1, y1, x2, y2], “label”: “motorcyclist”, “sub_label”: “wearing helmat” # or “not wearing helmat”}.</font>
>

![](检测pic.png)

```python
Qwen2.5-VL
[
	{"bbox_2d": [341, 258, 397, 360], "label": "motorcyclist", "sub_label": "not wearing helmat "},
	{"bbox_2d": [212, 332, 274, 448], "label": "motorcyclist", "sub_label": "not wearing helmat "},
	{"bbox_2d": [66, 124, 112, 198], "label": "motorcyclist", "sub_label": "not wearing helmat "},
	{"bbox_2d": [5, 235, 63, 320], "label": "motorcyclist", "sub_label": "wearing helmat "}
]
```





+ 更简洁高效的视觉编码器

视觉编码器在多模态大模型中扮演着至关重要的角色。我们<font style="color:#DF2A3F;">从头开始训练了一个原生动态分辨率的 ViT</font>，包括 CLIP、视觉-语言模型对齐和端到端训练等阶段。为了解决多模态大模型在训练和测试阶段 ViT 负载不均衡的问题，我们引入了<font style="color:#DF2A3F;">窗口注意力机制</font>，有效减少了 ViT 端的计算负担。在我们的 ViT 设置中，只有四层是全注意力层，其余层使用窗口注意力。最大窗口大小为 8x8，小于 8x8 的区域不需要填充，而是保持原始尺度，确保模型保持原生分辨率。此外，<font style="color:#DF2A3F;">为了简化整体网络结构，我们使 ViT 架构与 LLMs 更加一致，采用了 RMSNorm 和 SwiGLU 结构</font>。



对视觉语言模型进行了全面的评估，比较了 SOTA 模型以及同尺寸规模模型中表现最好的模型。在旗舰模型 Qwen2.5-VL-72B-Instruct 的测试中，它在一系列涵盖多个领域和任务的基准测试中表现出色，包括大学水平的问题、数学、文档理解、视觉问答、视频理解和视觉 Agent。值得注意的是，Qwen2.5-VL 在理解文档和图表方面具有显著优势，并且能够作为视觉 Agent 进行操作，而无需特定任务的微调。

![](qwen2.5性能1.png)

在较小的模型方面，Qwen2.5-VL-7B-Instruct 在多个任务中超越了 GPT-4o-mini，而 Qwen2.5-VL-3B 作为端侧 AI 的潜力股，甚至超越了我们之前版本 Qwen2-VL 的 7B 模型。

![](qwen2.5性能2.png)

![](qwen2.5性能3.png)





> 总结：不得不说，Qwen2.5-vl是真的强，在一些多模态任务上绰绰有余，而且检测任务上都表现很好。
>







