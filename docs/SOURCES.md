# 方法、数据与代码来源

研究依据：Guangtai Wang、Kechen Lin、Meng Han，Prototype Contrastive Learning for Heterogeneous Blockchain Federated Learning。现有依据为项目保存的论文稿件；这里不补写尚未核实的发表或录用信息。

补充实现基于2026年9月10日的独立NumPy机制实验。只实现语义原型加权表示协作和小样本原型收缩，使用本机两节点Geth合约约束轮次与提交。代码不是原论文完整官方实现。

历史统计来自main_20260910T103853Z_3428e9及其audit_20260910T105109Z_49a3a8复核。发布时的新集成验证不计入历史三种子统计。

原始数据：[EMNIST Balanced（NIST）](https://www.nist.gov/itl/products-and-services/emnist-dataset)。仓库提供抽样与划分程序、样本索引和源文件摘要，使用者从原来源获取图像。数据及第三方研究的权利归各自权利人，不受本仓库代码许可证重新授权。

本次改动包括解除对状态约束数字身份验证工程目录的依赖、工具路径配置、显式指定数据目录及发布说明；原始与发布文件摘要见source-manifest.json。
