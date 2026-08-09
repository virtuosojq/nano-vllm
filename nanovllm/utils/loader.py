import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认加载器: 整块拷贝(用于 replicated 权重, 每卡相同)。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从目录加载 HF 格式的 safetensors 权重到模型。
    核心思路:
      1. 每个 nn.Parameter 都挂了一个 weight_loader 钩子(在 LinearBase 构造时设置)
      2. loader 只负责"找到参数 + 调它的 weight_loader", 切分逻辑由各 Linear 类自己实现
      3. packed_modules_mapping 处理 q/k/v→qkv_proj、gate/up→gate_up_proj 的合并
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})   # 取 packed 映射表
    for file in glob(os.path.join(path, "*.safetensors")):                  # 遍历所有权重分片文件
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():                                    # 逐个权重
                # 匹配 packed 前缀: 如 q_proj/k_proj/v_proj 都合并到 qkv_proj
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)              # q_proj → qkv_proj
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")     # 取该参数自带的切分函数
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 非 packed 权重: 直接按 param 名取(若参数自带 weight_loader 则用它, 否则整块拷)
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
