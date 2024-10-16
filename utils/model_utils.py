import torch
import torch.nn as nn
import gc
import wandb 

from utils.eval_utils import parse_config
from models.resnet import BasicBlock, Bottleneck, resnet18, resnet50
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer
import numpy as np 
import matplotlib.pyplot as plt 


CONFIG_PATH = '/content/ptmq_log/config/gpu_config.yaml'
cfg = parse_config(CONFIG_PATH)

wandb.init(
    # set the wandb project where this run will be logged
    project="ptmq-pytorch(양자화된 weight, fp32 모델의 weight)-fp32-bin100",
    # track hyperparameters and run metadata
    config={
        "architecture": "ResNet-18",
        "dataset": "Imagenet",
        "recon_iters": cfg.quant.recon.iters,
    }
)

def log_feature_distribution(feature_map, model_name, bit_width):
    feature_flat = feature_map.cpu().numpy().flatten()

    q_min=0
    q_max=2**bit_width-1

    counts = np.zeros(q_max-q_min+1, dtype=int)

    overflow = 0

    underflow = 0
    for val in feature_flat:
        if q_min <= val <= q_max:
            counts[int(val)]+=1

        elif val>q_max:
            overflow +=1 
        elif val<q_min:
            underflow +=1
    bins = np.arange(q_min, q_max+2)
    plt.figure(figsize=(10,6))
    plt.bar(bins, counts, color='blue', edgecolor = 'black')
    plt.title(f"{model_name} {bit_width}bit - Feature Map Distribution")
    plt.xlabel("Quantized Value")
    plt.ylabel("Count")

        # wandb에 기록
    wandb.log({f"{model_name} {bit_width}bit - feature map distribution (bar)": wandb.Image(plt)})
    plt.close()


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    qoutput:    whether to quantize the block output
    out_mode:   setting inference block output mode
        - "mixed":  mixed feature output. used only for block reconstruction
        - "low":    low bit feature output
        - "med":    medium bit feature output
        - "high":   high bit feature output
    """
    def __init__(self, orig_module: BasicBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(orig_module.conv1, orig_module.relu1, config)
        self.conv2 = QuantizedLayer(orig_module.conv2, None, config, qoutput=False)
        if orig_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(orig_module.downsample[0], None, config, qoutput=False)
        self.activation = orig_module.relu2
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            
            self.f_l = None
            self.f_m = None
            self.f_h = None
            
            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3
            
            self.mixed_p = config.quant.ptmq.mixed_p
            
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(out)
                self.f_m = self.block_post_act_fake_quantize_med(out)
                self.f_h = self.block_post_act_fake_quantize_high(out)
                
                f_lmh = self.lambda1 * self.f_l + self.lambda2 * self.f_m + self.lambda3 * self.f_h
                f_mixed = torch.where(torch.rand_like(out) < self.mixed_p, out, f_lmh)
                out = f_mixed
                log_feature_distribution(out, "mixed", 32)
            elif self.out_mode == "low":
                out = self.block_post_act_fake_quantize_low(out)
                log_feature_distribution(out, "low", 3)
            elif self.out_mode == "med":
                out = self.block_post_act_fake_quantize_med(out)
                log_feature_distribution(out, "low", 4)

            elif self.out_mode == "high":
                out = self.block_post_act_fake_quantize_high(out)
                log_feature_distribution(out, "low", 5)

            else:
                raise ValueError(f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported")
        return out


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, Resnet-101, and ResNet-152.
    """
    def __init__(self, orig_module, config, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(orig_module.conv1, orig_module.relu1, config)
        self.conv2_relu = QuantizedLayer(orig_module.conv2, orig_module.relu2, config)
        self.conv3 = QuantizedLayer(orig_module.conv3, None, config, qoutput=False)
        
        if orig_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(orig_module.downsample[0], None, config.quant.w_qconfig, config.quant.a_qconfig, qoutput=False)
        self.activation = orig_module.relu3
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config.quant.a_qconfig)
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
    
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
            out_low = self.block_post_act_fake_quantize_low(out)
            out_med = self.block_post_act_fake_quantize_med(out)
            out_high = self.block_post_act_fake_quantize_high(out)
        return out

quant_modules = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck
}


def load_model(config):
    config['kwargs'] = config.get('kwargs', dict())
    model = eval(config['type'])(**config['kwargs'])
    checkpoint = torch.load(config.path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def set_qmodel_block_aqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            # print(name)
            module.out_mode = out_mode