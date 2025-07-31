# IPv6地址生成器 - 基于大语言模型的智能IPv6地址生成

## 项目概述

本项目基于Qwen 2.5-0.5B大语言模型，通过微调训练实现了智能的IPv6地址生成功能。给定网络前缀，模型能够生成该前缀下符合EUI-64格式的IPv6地址，用于网络扫描、安全测试等应用场景。

## 核心特性

- ✅ **基于前缀生成**：给定网络前缀，生成对应的IPv6地址
- ✅ **EUI-64格式支持**：生成符合EUI-64标准的IPv6地址
- ✅ **高成功率**：测试显示100%的地址生成成功率
- ✅ **大规模数据训练**：基于近300万个真实IPv6地址数据
- ✅ **结构化分析**：深度分析IPv6地址的网络前缀和接口标识符模式

## 项目结构

```
IPv6-Generator/
├── README.md                    # 项目说明文档
├── ipv6_generator.py           # 主要的IPv6地址生成器脚本
├── test_ipv6_generator.py      # 测试脚本
├── ieee/                       # 原始IEEE IPv6地址数据
│   ├── list_2001.txt          # 2001前缀的IPv6地址列表
│   ├── list_2409.txt          # 2409前缀的IPv6地址列表
│   ├── list_*.txt             # 其他前缀的IPv6地址列表
│   └── result.txt             # 汇总的地址统计信息
├── ipv6_generator_model/       # 训练好的模型文件（训练后生成）
│   ├── config.json            # 模型配置
│   ├── model.safetensors      # 模型权重
│   ├── tokenizer.json         # 分词器配置
│   └── ...                    # 其他模型文件
└── requirements.txt            # Python依赖包列表
```

## 环境要求

### 硬件要求
- **GPU**: 推荐8GB以上显存（支持CUDA）
- **内存**: 16GB以上RAM
- **存储**: 5GB以上可用空间

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（GPU训练）

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd IPv6-Generator
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证环境
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练（推荐）
python ipv6_generator.py --action train --sample-size 2000 --epochs 2

# 自定义训练参数
python ipv6_generator.py --action train \
    --model-name Qwen/Qwen2.5-0.5B \
    --data-dir ieee \
    --sample-size 5000 \
    --epochs 3 \
    --output-dir ./my_ipv6_model
```

**训练参数说明：**
- `--sample-size`: 训练样本数量（建议1000-5000）
- `--epochs`: 训练轮数（建议1-3轮）
- `--model-name`: 基础模型名称
- `--output-dir`: 模型保存路径

### 2. 生成IPv6地址

```bash
# 为指定前缀生成地址
python ipv6_generator.py --action generate \
    --prefix "2001:db8:85a3:0" \
    --num-addresses 5

# 使用自定义模型
python ipv6_generator.py --action generate \
    --prefix "2409:8a3c:1af0:1e01" \
    --num-addresses 3 \
    --output-dir ./my_ipv6_model
```

### 3. 评估模型性能

```bash
# 使用默认测试前缀评估
python ipv6_generator.py --action evaluate

# 使用自定义模型评估
python ipv6_generator.py --action evaluate --output-dir ./my_ipv6_model
```

### 4. 交互式测试

```bash
# 运行测试脚本
python test_ipv6_generator.py

# 交互式模式
python test_ipv6_generator.py --interactive
```

## 数据说明

### IEEE地址数据
- **来源**: IEEE分配的IPv6地址数据
- **格式**: EUI-64格式的IPv6地址
- **规模**: 约296万个地址，覆盖2569个网络前缀
- **结构**: 网络前缀(64位) + 接口标识符(64位)

### 训练数据格式
训练时，原始地址被转换为以下格式：
```
Network: 2001:1284:f502:1034 -> Address: 2001:1284:f502:1034:665e:10ff:fed5:3649
EUI64: 2001:1284:f502:1034 -> Interface: 665e:10ff:fed5:3649 -> Address: 2001:1284:f502:1034:665e:10ff:fed5:3649
```

## 技术原理

### 1. 模型架构
- **基础模型**: Qwen 2.5-0.5B (494M参数)
- **训练方式**: 微调（Fine-tuning）
- **任务类型**: 条件文本生成

### 2. IPv6地址结构理解
```
IPv6地址结构: [网络前缀:64位] + [接口标识符:64位]
示例: 2001:1284:f502:1034:665e:10ff:fed5:3649
     |-- 网络前缀 --|  |-- 接口标识符 --|
```

### 3. EUI-64格式特征
- 包含特征字符串 "ff:fe"
- 由MAC地址转换而来
- 符合IEEE标准

## 性能指标

### 训练结果
- **训练样本**: 1000-5000个（可配置）
- **训练时间**: 约10-30分钟（取决于样本数量和硬件）
- **模型大小**: 约1GB

### 生成性能
- **成功率**: 100%（测试结果）
- **前缀匹配率**: 100%
- **地址有效性**: 100%
- **生成速度**: 每个地址约1-3秒

### 测试结果示例
```
=== 测试前缀: 2001:db8:85a3:0 ===
✓ 生成地址 1: 2001:db8:85a3:0:66b6:2cff:fe45:7267
✓ 生成地址 2: 2001:db8:85a3:0:66a7:5cff:fe25:3719
✓ 生成地址 3: 2001:db8:85a3:0:621:6aff:fe76:3113

总体测试结果: 成功率 100.00%
```

## 应用场景

1. **网络扫描**: 为IPv6网络扫描生成候选地址
2. **安全测试**: 渗透测试中的IPv6地址枚举
3. **网络规划**: 辅助网络管理员进行IPv6地址规划
4. **研究用途**: IPv6地址分布和模式研究

## 注意事项

### 训练注意事项
1. **显存要求**: 建议8GB以上GPU显存
2. **训练时间**: 根据样本数量调整，避免过拟合
3. **数据质量**: 确保IEEE数据文件完整性

### 使用注意事项
1. **网络前缀格式**: 必须是有效的IPv6前缀格式
2. **生成地址用途**: 仅用于合法的网络测试和研究
3. **伦理使用**: 遵守网络安全和隐私相关法律法规

## 故障排除

### 常见问题

**Q: 训练时出现CUDA内存不足**
```bash
# 解决方案：减少batch size或使用CPU训练
python ipv6_generator.py --action train --sample-size 1000
```

**Q: 模型生成的地址格式不正确**
```bash
# 解决方案：检查训练数据质量，重新训练
python ipv6_generator.py --action train --epochs 2
```

**Q: 生成速度过慢**
```bash
# 解决方案：使用GPU加速或减少生成数量
python ipv6_generator.py --action generate --prefix "2001:db8::" --num-addresses 3
```

## 开发计划

- [ ] 支持更多IPv6地址格式（非EUI-64）
- [ ] 添加地址活跃性预测功能
- [ ] 优化模型大小和推理速度
- [ ] 支持批量前缀处理
- [ ] 添加Web界面

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目维护者: [drnbnonono]
- 邮箱: [2116197323@qq.com]

## 致谢

- Qwen团队提供的优秀基础模型
- IEEE提供的IPv6地址数据
- Hugging Face提供的模型训练框架
