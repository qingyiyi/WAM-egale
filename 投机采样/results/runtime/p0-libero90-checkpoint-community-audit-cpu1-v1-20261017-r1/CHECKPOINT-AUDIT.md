# LIBERO90 同代 LingBot-VA 社区 checkpoint 核查

本次有界公开元数据核查未确认可直接使用的同代 LingBot-VA LIBERO90 target。结论是“未核实”，不是“绝对不存在”。未下载权重、未加载模型、未做GPU/SSH/测试/训练或自行SHA。四权限保持false。

|类别|实际观察|能否作为90 target|
|---|---|---|
|同代且可核实90训练|本次未找到明确训练来源和绑定评测的权重|UNKNOWN，不能启动依赖它的质量链|
|rbehnam19六个checkpoint|每个均有10,177,831,668B transformer safetensors、449B config；无README|训练suite/normalization/评测UNKNOWN|
|Wjjjh Spatial/Object/Goal|三套明确各10task社区模型，文件与训练/评测说明齐|不是90；本地loader和质量未验收|
|bazaar-research / XinY0201 handoff|前者代码/bundle，后者Long SFT交接；清单无具名模型权重|不能当90现成target|
|peter-chl FastWAM90|有.pt及90评测路径；README为Wan22→ActionDiT路线|不同架构，不能替换LingBot-VA|

## 固定 publisher revisions 与文件

以下commit来自HF元数据，不是本地计算的文件hash。每仓库只列最多5个权重路径；详细配置和出处见RESULT.json。

|repo|publisher commit|权重文件（前5）|
|---|---|---|
|[rbehnam19/lingbot-va-checkpoint-4000](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-4000/tree/94ddecb7081519063a2958295ec03f25e3ad3b2a)|94ddecb7081519063a2958295ec03f25e3ad3b2a|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[rbehnam19/lingbot-va-checkpoint-5000](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-5000/tree/af11b770ff1ba02a712618046bdeb304916564cc)|af11b770ff1ba02a712618046bdeb304916564cc|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[rbehnam19/lingbot-va-checkpoint-15000](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-15000/tree/a006739ce7001d05360be4e9db93b4796038a39a)|a006739ce7001d05360be4e9db93b4796038a39a|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[rbehnam19/lingbot-va-checkpoint-17500](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-17500/tree/f9ea51752a12c8efcd278735fc63a17e8a77fa27)|f9ea51752a12c8efcd278735fc63a17e8a77fa27|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[rbehnam19/lingbot-va-checkpoint-20000](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-20000/tree/61a29ed4a3873568caa6871276b7212602079bb2)|61a29ed4a3873568caa6871276b7212602079bb2|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[rbehnam19/lingbot-va-checkpoint-30000](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-30000/tree/40be4c4f7b12a148d00cbb8adce500d8f4aef784)|40be4c4f7b12a148d00cbb8adce500d8f4aef784|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[bazaar-research/lingbot-va](https://huggingface.co/bazaar-research/lingbot-va/tree/039f87d1df036cecffeda13444e5209f3aaba5e1)|039f87d1df036cecffeda13444e5209f3aaba5e1|无具名权重文件|
|[XinY0201/lingbot-va-official-sft-handoff](https://huggingface.co/XinY0201/lingbot-va-official-sft-handoff/tree/90d9bad6c7500c95e09e533c2bfed7045b831dd5)|90d9bad6c7500c95e09e533c2bfed7045b831dd5|无具名权重文件|
|[Wjjjh/lingbot-va-libero-object](https://huggingface.co/Wjjjh/lingbot-va-libero-object/tree/489806feb4074eee71f97180948e8b85e33ebeda)|489806feb4074eee71f97180948e8b85e33ebeda|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[Wjjjh/lingbot-va-libero-goal](https://huggingface.co/Wjjjh/lingbot-va-libero-goal/tree/42455fd87bc5b9c4613cc13b03be2f2d80605efe)|42455fd87bc5b9c4613cc13b03be2f2d80605efe|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[Wjjjh/lingbot-va-libero-spatial](https://huggingface.co/Wjjjh/lingbot-va-libero-spatial/tree/933ab666f8cd21ff71d8765dea9fdd110512034f)|933ab666f8cd21ff71d8765dea9fdd110512034f|transformer/diffusion_pytorch_model.safetensors (10177831668B)|
|[peter-chl/fastwam-sft-libero90](https://huggingface.co/peter-chl/fastwam-sft-libero90/tree/56229b1a6e9a0c8f977d3386ad5bcc415ac8d335)|56229b1a6e9a0c8f977d3386ad5bcc415ac8d335|l90push_lora_step10_trainable.pt (2122929190B); lora_90long_sft500.pt (12041735545B); rl_run50_step50.pt (12041973541B); round24_step_002000.pt (12041735545B); round26_strongaug_step_001500.pt (12041735545B)|

## 配置与社区证据

六个未标suite的rbehnam19模型及Wjjjh三套配置均为WanTransformer3DModel，30层、24heads、head_dim128、FFN14336、text4096、in/out48、patch[1,2,2]；这些已列字段与本地Long config一致。社区config另显式action_dim30，对应本地model.py:826–841默认值。发布attn_mode=flex，不能因此继承现有torch/O3的数值、KV或质量证书。未读取safetensors内容。

Wjjjh三套说明均从robbyant/lingbot-va-base训练5000 updates，500episode、suite-specific动作分位数；仅发布transformer，另需base的VAE、tokenizer和text encoder。publisher声称Object1486/1500(99.07%)、Goal1451/1500(96.73%)、Spatial1430/1500(95.33%)；这些是社区自报，未由我们复现。Object/Goal第一轮seed未记录；Spatial说明旧数据导出不兼容后重新训练。每次评测10task×50episode、horizon800、双128相机，不能泛称90task结果。

证据固定链接：
- [bazaar-research/lingbot-va固定模型卡](https://huggingface.co/bazaar-research/lingbot-va/resolve/039f87d1df036cecffeda13444e5209f3aaba5e1/README.md)
- [XinY0201/lingbot-va-official-sft-handoff固定模型卡](https://huggingface.co/XinY0201/lingbot-va-official-sft-handoff/raw/90d9bad6c7500c95e09e533c2bfed7045b831dd5/README.md)
- [Wjjjh/lingbot-va-libero-object固定模型卡](https://huggingface.co/Wjjjh/lingbot-va-libero-object/resolve/489806feb4074eee71f97180948e8b85e33ebeda/README.md)
- [Wjjjh/lingbot-va-libero-goal固定模型卡](https://huggingface.co/Wjjjh/lingbot-va-libero-goal/raw/42455fd87bc5b9c4613cc13b03be2f2d80605efe/README.md)
- [Wjjjh/lingbot-va-libero-spatial固定模型卡](https://huggingface.co/Wjjjh/lingbot-va-libero-spatial/raw/933ab666f8cd21ff71d8765dea9fdd110512034f/README.md)
- [peter-chl/fastwam-sft-libero90固定模型卡](https://huggingface.co/peter-chl/fastwam-sft-libero90/resolve/56229b1a6e9a0c8f977d3386ad5bcc415ac8d335/README.md)

## 搜索边界与下一步

HF模型名搜索lingbot-va返回68项；libero90返回46项。libero_90/libero-90两次均触及100项返回上限，因此不据后两者宣称全站不存在。候选API完整siblings只用于过滤；没有扫描权重树或下载权重。FastWAM配置在固定commit的resolve/raw均HTTP401，保留UNKNOWN和失败记录。

继续Long准备有现成target依据；90工作仍缺同代target身份与90训练来源。tiny draft重训不能替代target本身在90task上的能力和动作归一化。若之后取得明确90target，再登记teacher数据、draft FIT/CAL和完整配对评测；本次未启动任何下载、迁移或训练。
