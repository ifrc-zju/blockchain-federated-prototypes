# 区块链协作的联邦原型学习

本项目实现区块链协作的联邦原型学习，将跨客户端语义加权对比学习和小样本原型收缩接入轮次与提交状态校验流程。客户端使用NumPy完成本地训练，智能合约记录轮次、客户端提交及模型／原型摘要；链下程序核对提交文件与链上状态后完成聚合。

**贡献者：** 潘启颖、刘宇、王者、韩蒙

[方法与实验报告](docs/technical-report.md) · [实验数据](data/published-results/) · [运行验证](docs/VALIDATION.md)

训练采用小型全连接网络和EMNIST Balanced数据，两节点Geth测试链在本机运行。可分别运行基线、普通监督对比、语义加权对比、原型收缩及组合方法，比较各机制在相同数据划分下的表现。

## 方法与代码

- `tools/learning.py`：语义加权原型对比学习、梯度、原型收缩与聚合。
- `tools/PrototypeRegistry.sol`：任务、客户端、轮次、先验及提交集合约束。
- `tools/chain_transport.py`：两节点测试链、交易提交与状态读取。
- `tools/independent_audit.py`：独立重算训练、聚合、预测与历史链状态。

## 安装与数据准备

需要 Python 3.12、Geth **1.13.15** 和 solc **0.8.24**。将 geth、solc 加入 PATH，或设置 `GETH_BINARY`、`SOLC_BINARY` 为可执行文件的完整路径。Geth 必须使用上述支持 Clique 的版本；新版已移除该出块方式，见[官方说明](https://geth.ethereum.org/docs/fundamentals/private-network)。程序只连接本机新建测试链，不连接主网。

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 设置示例：

```powershell
$env:GETH_BINARY = "C:/tools/geth.exe"
$env:SOLC_BINARY = "C:/tools/solc.exe"
```

原始数据使用[NIST发布的EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset)。下载并解压IDX格式数据，指定包含以下文件的目录：

- `emnist-balanced-{train,test}-images-idx3-ubyte`
- `emnist-balanced-{train,test}-labels-idx1-ubyte`
- `emnist-balanced-mapping.txt`

```bash
python tools/prepare_input.py --data-dir "/path/to/emnist_balanced"
python tools/test_learning.py
python tools/run_experiment.py --smoke
python tools/independent_audit.py <上一步输出的套件编号>
```

准备脚本按种子2300抽取4,700张训练图像和1,880张测试图像；按种子2301、2302、2303划分6个客户端。首次运行创建`I23-INPUT`，不会覆盖已有输入。输入目录记录本机原始数据路径，若移机应重新生成。测试链默认RPC端口19623、19624，独立复核使用19625。

`--smoke`仅运行一个种子的WS组、初始化和1轮训练；去掉该参数运行5组×3种子×10轮的完整配置。每次运行生成新测试账户，不需要真实钱包或主网资金。链库、私钥和本机运行目录不上传。

## 数据与结论范围

公开历史统计包含逐轮、逐种子及成对比较结果。重建脚本、样本索引和划分说明支持使用者自行生成输入；原始图像从NIST获取。

既有小规模实验中WPC与原型收缩均未表现出准确率优势，具体结果与配置限制见[实验报告](docs/technical-report.md)。运行验证检查指定配置下的执行过程和独立数值复核，不用于推断性能提升。

## 研究来源与贡献

本项目依据Guangtai Wang、Kechen Lin、Meng Han的论文《Prototype Contrastive Learning for Heterogeneous Blockchain Federated Learning》中的部分机制独立实现。实现范围为语义原型加权表示协作和小样本原型收缩，未包含IRD、MHN及原论文完整模型，因此不构成原论文完整训练代码或性能复现。公式选择、代码与数据来源见[来源说明](docs/SOURCES.md)。

## 许可

仓库自有代码及配套使用说明采用[MIT许可](LICENSE)。技术报告、研究论文和第三方数据保留各自署名及权利；MIT许可不替代其原有授权。
