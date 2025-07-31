#!/usr/bin/env python3
"""
IPv6地址生成器 - 基于前缀的智能地址生成
核心思想：给定网络前缀，生成该前缀下可能活跃的IPv6地址
"""

import torch
import json
import random
import ipaddress
from pathlib import Path
from collections import defaultdict, Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
import re

class IPv6AddressAnalyzer:
    """IPv6地址结构分析器"""
    
    def __init__(self):
        self.prefix_patterns = defaultdict(list)
        self.eui64_patterns = []
        self.interface_patterns = defaultdict(list)
        
    def analyze_addresses(self, data_dir="ieee"):
        """分析IPv6地址结构特点"""
        print("分析IPv6地址结构...")
        
        data_path = Path(data_dir)
        total_addresses = 0
        
        for file_path in data_path.glob("list_*.txt"):
            print(f"分析文件: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    addr = line.strip()
                    if addr:
                        self._analyze_single_address(addr)
                        total_addresses += 1
        
        print(f"总共分析了 {total_addresses} 个地址")
        self._print_analysis_results()
        
        return self.prefix_patterns
    
    def _analyze_single_address(self, addr):
        """分析单个地址"""
        try:
            # 验证地址有效性
            ipaddress.IPv6Address(addr)
            
            parts = addr.split(':')
            
            # 网络前缀分析 (前64位，通常是前4段)
            if len(parts) >= 4:
                network_prefix = ':'.join(parts[:4])
                self.prefix_patterns[network_prefix].append(addr)
            
            # EUI-64格式检测
            if 'ff:fe' in addr.lower():
                self.eui64_patterns.append(addr)
                
                # 提取接口标识符模式 (后64位)
                if len(parts) >= 4:
                    interface_id = ':'.join(parts[4:])
                    self.interface_patterns[network_prefix].append(interface_id)
            
        except ipaddress.AddressValueError:
            pass
    
    def _print_analysis_results(self):
        """打印分析结果"""
        print(f"\n=== IPv6地址结构分析结果 ===")
        print(f"发现网络前缀数量: {len(self.prefix_patterns)}")
        print(f"EUI-64格式地址数量: {len(self.eui64_patterns)}")
        
        # 显示最常见的前缀
        prefix_counts = {prefix: len(addrs) for prefix, addrs in self.prefix_patterns.items()}
        top_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n前10个最常见的网络前缀:")
        for prefix, count in top_prefixes:
            print(f"  {prefix}: {count} 个地址")

class IPv6Generator:
    """IPv6地址生成器"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.analyzer = IPv6AddressAnalyzer()
        
    def load_model(self):
        """加载预训练模型"""
        print(f"加载模型: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=None
            )
            
            self.model = self.model.to(self.device)
            print(f"模型加载成功，参数量: {self.model.num_parameters():,}")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def prepare_training_data(self, data_dir="ieee", sample_size=5000):
        """准备基于前缀的训练数据"""
        print("准备基于前缀的训练数据...")
        
        # 分析地址结构
        prefix_patterns = self.analyzer.analyze_addresses(data_dir)
        
        training_samples = []
        
        # 为每个网络前缀创建训练样本
        for network_prefix, addresses in prefix_patterns.items():
            if len(addresses) < 3:  # 跳过样本太少的前缀
                continue
            
            # 随机采样该前缀下的地址
            sample_count = min(20, len(addresses))  # 每个前缀最多20个样本
            sampled_addresses = random.sample(addresses, sample_count)
            
            for addr in sampled_addresses:
                # 创建前缀到完整地址的映射
                training_text = f"Network: {network_prefix} -> Address: {addr}"
                training_samples.append(training_text)
                
                # 如果是EUI-64格式，额外创建EUI-64特定的训练样本
                if 'ff:fe' in addr.lower():
                    parts = addr.split(':')
                    if len(parts) >= 4:
                        interface_id = ':'.join(parts[4:])
                        eui64_text = f"EUI64: {network_prefix} -> Interface: {interface_id} -> Address: {addr}"
                        training_samples.append(eui64_text)
        
        print(f"创建了 {len(training_samples)} 个训练样本")
        
        # 限制总样本数
        if len(training_samples) > sample_size:
            training_samples = random.sample(training_samples, sample_size)
            print(f"采样到 {sample_size} 个训练样本")
        
        # 分割训练和验证集
        random.shuffle(training_samples)
        split_idx = int(len(training_samples) * 0.9)
        
        return training_samples[:split_idx], training_samples[split_idx:]
    
    def create_dataset(self, texts, max_length=128):
        """创建训练数据集"""
        print("创建数据集...")
        
        tokenized_data = []
        for text in texts:
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized_data.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze()
            })
        
        dataset = Dataset.from_dict({
            'input_ids': [item['input_ids'] for item in tokenized_data],
            'attention_mask': [item['attention_mask'] for item in tokenized_data]
        })
        
        return dataset
    
    def train(self, train_dataset, val_dataset, output_dir="./ipv6_generator_model", epochs=3):
        """训练模型"""
        print("开始训练IPv6地址生成器...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            fp16=False,
            learning_rate=3e-5,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        try:
            trainer.train()
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            print(f"模型训练完成，保存到: {output_dir}")
            return True
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def generate_addresses(self, network_prefix, num_addresses=5, model_path=None):
        """基于网络前缀生成IPv6地址"""
        if model_path:
            # 加载训练好的模型
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            model = model.to(self.device)
        else:
            tokenizer = self.tokenizer
            model = self.model
        
        if model is None:
            print("错误: 模型未加载")
            return []
        
        model.eval()
        generated_addresses = []
        
        with torch.no_grad():
            for i in range(num_addresses):
                # 创建提示词
                prompt = f"Network: {network_prefix} -> Address:"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_part = generated_text[len(prompt):].strip()
                
                # 提取IPv6地址
                ipv6_addr = self._extract_ipv6_address(generated_part)
                if ipv6_addr and ipv6_addr.startswith(network_prefix):
                    generated_addresses.append(ipv6_addr)
                    print(f"✓ 生成地址: {ipv6_addr}")
                else:
                    print(f"✗ 生成失败: {generated_part}")
        
        return generated_addresses
    
    def _extract_ipv6_address(self, text):
        """从文本中提取IPv6地址"""
        # IPv6地址正则表达式
        pattern = r'[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}'
        
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                ipaddress.IPv6Address(match)
                return match
            except:
                continue
        
        return None
    
    def evaluate(self, test_prefixes=None, model_path=None):
        """评估生成器性能"""
        if test_prefixes is None:
            test_prefixes = [
                "2001:1284:f502:1034",
                "2409:8a3c:1af0:1e01", 
                "2a01:4f8:c17:b8f",
                "2400:cb00:2048:1"
            ]
        
        print("=== 评估IPv6地址生成器 ===")
        
        total_generated = 0
        valid_generated = 0
        
        for prefix in test_prefixes:
            print(f"\n测试前缀: {prefix}")
            addresses = self.generate_addresses(prefix, num_addresses=3, model_path=model_path)
            
            total_generated += 3  # 尝试生成3个
            valid_generated += len(addresses)
            
            if addresses:
                print(f"成功生成 {len(addresses)} 个地址:")
                for addr in addresses:
                    print(f"  {addr}")
            else:
                print("未生成有效地址")
        
        success_rate = valid_generated / total_generated if total_generated > 0 else 0
        print(f"\n=== 评估结果 ===")
        print(f"总尝试生成: {total_generated}")
        print(f"成功生成: {valid_generated}")
        print(f"成功率: {success_rate:.2%}")
        
        return success_rate

def main():
    parser = argparse.ArgumentParser(description='IPv6地址生成器')
    parser.add_argument('--action', choices=['train', 'generate', 'evaluate'], default='train', help='执行的操作')
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-0.5B', help='预训练模型名称')
    parser.add_argument('--data-dir', default='ieee', help='数据目录')
    parser.add_argument('--sample-size', type=int, default=3000, help='训练样本大小')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--output-dir', default='./ipv6_generator_model', help='模型输出目录')
    parser.add_argument('--prefix', help='生成地址时使用的网络前缀')
    parser.add_argument('--num-addresses', type=int, default=5, help='生成地址数量')
    
    args = parser.parse_args()
    
    generator = IPv6Generator(args.model_name)
    
    if args.action == 'train':
        print("=== 训练IPv6地址生成器 ===")
        
        if not generator.load_model():
            return
        
        train_texts, val_texts = generator.prepare_training_data(args.data_dir, args.sample_size)
        train_dataset = generator.create_dataset(train_texts)
        val_dataset = generator.create_dataset(val_texts)
        
        success = generator.train(train_dataset, val_dataset, args.output_dir, args.epochs)
        
        if success:
            print("训练完成，开始评估...")
            generator.evaluate(model_path=args.output_dir)
    
    elif args.action == 'generate':
        if not args.prefix:
            print("错误: 生成地址需要指定 --prefix 参数")
            return
        
        print(f"=== 为前缀 {args.prefix} 生成IPv6地址 ===")
        
        if Path(args.output_dir).exists():
            addresses = generator.generate_addresses(
                args.prefix, 
                args.num_addresses, 
                model_path=args.output_dir
            )
            
            if addresses:
                print(f"\n生成的地址:")
                for addr in addresses:
                    print(addr)
            else:
                print("未生成有效地址")
        else:
            print(f"错误: 模型路径 {args.output_dir} 不存在")
    
    elif args.action == 'evaluate':
        print("=== 评估IPv6地址生成器 ===")
        
        if Path(args.output_dir).exists():
            generator.evaluate(model_path=args.output_dir)
        else:
            print(f"错误: 模型路径 {args.output_dir} 不存在")

if __name__ == "__main__":
    main()
