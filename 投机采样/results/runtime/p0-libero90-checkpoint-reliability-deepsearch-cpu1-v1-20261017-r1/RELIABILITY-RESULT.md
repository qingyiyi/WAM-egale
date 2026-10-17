# LIBERO90 同代 LingBot-VA 权重：独立可靠性深挖

终态：**COMPLETED_METADATA_ONLY_NO_VERIFIED_LIBERO90_TARGET**。截至北京时间2026-10-17 14:10，本次有界检索仍未找到能够核实“同代LingBot-VA＋LIBERO90训练＋对应归一化＋确切权重发布”的现成target。不能据此断言全网绝对不存在。

这是一份公开资产可用性审计，不改项目固定科研快照4f04b35、旧FAILED、旧NO_GO、默认配置或四项false。读取当前外部发布元数据只用于寻找资产，未用外部main覆盖本地科学判断。

## OBSERVED

1. **官方两个发布平台相互印证仍是三套同代VA模型。** HF官方作者列表仅base、RoboTwin、LIBERO-Long。ModelScope按[官方SDK只读列表接口](https://github.com/modelscope/modelscope/blob/v1.30.0/modelscope/hub/api.py#L863)完整返回Robbyant账号26/26模型，同代VA仍只有上述三项。原始响应见`hf_official_author.json`与`ms_official_owner_models_noslash.json`。

2. **官方代码没有另一条公开发布分支。** [7c6ffa9固定README](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/README.md)列相同三模型；API返回一个main分支、0 release、0 tag。23个LIBERO相关issue/PR及相关评论未给出90权重。特别是[#75后续评论](https://github.com/Robbyant/lingbot-va/issues/75#issuecomment-4405253065)仍在询问非Long权重；这条是社区提问，不能写成官方宣告其他权重不存在。

3. **rbehnam19六套权重仍无法证明是90。** 既有审计已固定六个发布revision，每套均有`transformer/diffusion_pytorch_model.safetensors`（10,177,831,668B）及449B配置；主要架构字段与本地共享主干一致。此次追其公开HF用户、模型、dataset、提交历史、同名GitHub和所属组织：HF无其dataset，提交仅通用上传消息，未见训练suite、action quantile或评测来源。同名GitHub无公开仓库，但与HF账号的现实身份未证实，不据此替发布者补身份。所属North-Star组织公开dataset实际标注BEHAVIOR-1K，不能用来证明六套模型训练内容。[30000固定发布](https://huggingface.co/rbehnam19/lingbot-va-checkpoint-30000/tree/40be4c4f7b12a148d00cbb8adce500d8f4aef784)只是具名文件存在证据。

4. **又排查了未标suite及改架构模型。** Brankozz DMD两端checkpoint、Unwinding arms有约10.18GB的同类权重，但未提供可绑定的90训练/normalizer/评测；不能按step或shape猜。`arashakb/Lingbot-VA-W4A8`模型卡明确是RoboTwin量化；`ZhuoranChen/lingbot-va-mot`明确是未微调的10B转换MoT；CoreAI为base量化导出，均不能冒充当前90 target。额外候选的完整路径、大小、revision、许可及UNKNOWN字段列在`RESULT.json`。

5. **大备份fanzh03也没有明确90身份。** [ee6fca8固定目录](https://huggingface.co/fanzh03/lingbot-va/tree/ee6fca8b6f39a0d3be3ee7884012a92fb7c30408)包含4suite、Long/Goal/Object/Spatial与shot实验，README仅28B。未见明确90发布身份；没有下载其大文件或把目录名当训练证明。

6. **近似搜索命中需要排除。** `aabyaneh/fasterwam`的train90命名来自每任务90/10 episode切分，公开任务为fruit_pick/place_cube/dish_utensil/push_t，不是LIBERO90。FastWAM90是另一架构。OpenWAM官网确有四个LIBERO90迁移任务，但[固定代码README](https://github.com/OpenWAM/OpenWAM/blob/030da766bbd882f2216606ec4c31a1dd224f82b3/README.md)明确policy checkpoint待发布，当前只有视频预训练权重，不能替换同代LingBot-VA。

## 最接近且元数据较完整的替代

|发布者模型|固定revision|训练与归一化|权重|可作90 target|
|---|---|---|---|---|
|[Wjjjh/Object](https://huggingface.co/Wjjjh/lingbot-va-libero-object/tree/489806feb4074eee71f97180948e8b85e33ebeda)|489806feb4074eee71f97180948e8b85e33ebeda|Object10；模型卡给对应q01/q99|同路径10,177,831,668B|否|
|[Wjjjh/Goal](https://huggingface.co/Wjjjh/lingbot-va-libero-goal/tree/42455fd87bc5b9c4613cc13b03be2f2d80605efe)|42455fd87bc5b9c4613cc13b03be2f2d80605efe|Goal10；模型卡给对应q01/q99|同路径10,177,831,668B|否|
|[Wjjjh/Spatial](https://huggingface.co/Wjjjh/lingbot-va-libero-spatial/tree/933ab666f8cd21ff71d8765dea9fdd110512034f)|933ab666f8cd21ff71d8765dea9fdd110512034f|Spatial10；模型卡给对应q01/q99|同路径10,177,831,668B|否|

三者具名文件均为`transformer/diffusion_pytorch_model.safetensors`，许可模型卡Apache-2.0；从官方base、套件对应500条示范训练5000更新，共享主干30层/24heads/head_dim128/FFN14336/action_dim30。另依赖base的VAE、text encoder与tokenizer。发布者自报三轮成功Object1486/1500、Goal1451/1500、Spatial1430/1500；这些没有由本项目复现，原seed缺项及训练资料问题保留，不能直接继承本项目质量证书。

## UNKNOWN / 停止理由

- 未发现同时绑定90训练suite、7维动作归一化、确切checkpoint和90评测协议的同代权重。teacher能力未知时，重训tiny draft不能代替teacher在90上的适配。
- HF名称检索返回68项，base衍生标签检索亦未出现90身份；搜索引擎、公开repo与标签不能覆盖私有、未标注或未索引资产。ModelScope官方账号列表完整，但社区全站搜索API返回404，不能宣称已穷举ModelScope社区。
- 没下载或加载任何候选，实际loader/数值/闭环可用性仍为UNKNOWN。模型名、safetensors存在和config相同都不等于90可用。
- 继续盲下多套10GB文件并跑成功率不能补齐发布身份，还会占用本轮评测资源；本次到元数据证据边界止损。

## RECOMMENDATION

继续已授权Long三臂完整评测。90 draft采集/训练/评测维持BLOCKED，直到拿到明确90 target及配套normalizer。若目标是尽快补齐常用四套LIBERO论文表，Wjjjh三个已具名的同代小套件是目前更可追溯的候选；需用户明确范围选择后再登记各套件draft/loader/完整评测，不自动把90请求替换成三套，也不启动第二环境。

## 资源与证据

CPU固定12号一核，约10.2分钟只读检索；GPU 0、SSH 0、权重下载0、模型加载0、自行SHA 0，无遗留自有作业。81条来源记录含重试与失败，69条明确HTTP200，其余失败/未定状态保留；不是81个独立模型。新增约1.5MiB，低于4MiB上限。

`RESULT.json`提供候选结构化身份与四false；`SOURCES.json`是原响应索引；`SEARCH-LOG.json`列检索边界和一手链接。前次12候选详细审计仍位于`../p0-libero90-checkpoint-community-audit-cpu1-v1-20261017-r1/RESULT.json`，未覆盖旧记录。
