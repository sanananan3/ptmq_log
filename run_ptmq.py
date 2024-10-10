import torch
import torch.nn as nn
import numpy as np

import time
import copy
import logging
import argparse
import wandb
import matplotlib.pyplot as plt
import utils
import utils.eval_utils as eval_utils
from utils.ptmq_recon import ptmq_reconstruction
from utils.fold_bn import search_fold_and_remove_bn, StraightThrough
from utils.model_utils import quant_modules, load_model, set_qmodel_block_aqbit
from quant.quant_state import enable_calib_without_quant, enable_quantization, disable_all
from quant.quant_module import QuantizedLayer, QuantizedBlock
from quant.fake_quant import QuantizeBase
from quant.observer import ObserverBase
from utils.eval_utils import DataSaverHook, StopForwardException, parse_config

logger = logging.getLogger('ptmq')


""" 
def log_weight_distribution(layer_weights, model_name, bit_width=None):

    layer의 weight 데이터를 wandb에 바 차트로 로깅하는 함수
    
    Args:
        layer_weights (torch.Tensor): 레이어의 weight 데이터
        model_name (str): 모델 이름 (FP32 또는 Quantized)

    if bit_width:
        weight = layer_weights.cpu().numpy().flatten()  # weight 데이터를 numpy로 변환하고 평탄화        

        q_min = 0
        q_max = 2 ** bit_width - 1  # for 3-bit, q_min=0 and q_max=7

        counts = np.zeros(q_max - q_min + 1, dtype=int)  # initialize counts array for bins 0 to 7
        underflow = 0  # q_min 미만의 값 카운트
        overflow = 0  # q_max 초과의 값 카운트

        for val in weight:
            if val < q_min:
                underflow += 1
            elif val > q_max:
                overflow += 1
            else:
                counts[int(val)] += 1

        # 막대그래프 그리기
        bins = np.arange(q_min, q_max + 1)  # 0 ~ 7 까지의 값
        plt.figure(figsize=(10, 6))
        plt.bar(bins, counts, color='blue', edgecolor='black')
        plt.title(f"{model_name} {bit_width}bit - weight distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Count")
        
        # Underflow와 Overflow 값 표시
        plt.text(q_min, max(counts) * 0.9, f"Underflow: {underflow}", fontsize=12, color='red')
        plt.text(q_max, max(counts) * 0.9, f"Overflow: {overflow}", fontsize=12, color='red')

        # 그래프를 wandb에 로깅
        wandb.log({f"{model_name} {bit_width}bit - weight distribution (bar)": wandb.Image(plt)})
        plt.close()

    else: 
        weight = layer_weights.cpu().numpy().flatten()
        num_bins=100

        plt.figure(figsize=(10,6))
        plt.hist(weight, bins=num_bins, color='blue', edgecolor='black')
        plt.title("FP32 - weight distribution")
        plt.xlabel("Weight value")
        plt.ylabel("Count")

        wandb.log({"FP32 - weight distribution-bins50": wandb.Image(plt)})
        plt.close()

        """

def get_quantized_value_histogram(output, bit_width):

    q_min= 0
    q_max = 2 ** bit_width -1 


    # 양자화 범위 내 값들의 발생 빈도 계산 

    hist = {i:0 for i in range (q_min, q_max+1)}

    hist['below_min'] = 0 
    hist['above_max'] = 0

    # 값 빈도 카운트 
    print("a_qbit", bit_width)
    layer_output_flatten = output.view(-1) # 텐서를 1D로 평탄화 
    print("평탄화는 시킴 ", layer_output_flatten)

    for value in layer_output_flatten:
        value_int = int(value.item()) # 텐서 값을 int 로 변환 
        if value_int < q_min:
            hist['below_min'] += 1
        elif value_int > q_max:
            hist['above_max'] += 1
        else: 
            hist[value_int] += 1

    print("hist는 return 돼따 ")
    return hist 


def log_histogram_to_wandb(hist, bit_width, iteration):
    """
        wandb에 히스토그램을 로깅하는 함수 
        iteration => 현재 iteration 수 
    """
    wandb.log({
        f"{bit_width} bit_quantized_value_histogram": wandb.Histogram(
            np_histogram = (list(hist.keys()), list(hist.values()))
        ),
        "iteration": iteration
    })

def quantize_model(model, config):
    def replace_module(module, config, qoutput=True):
        children = list(iter(module.named_children()))
        ptr, ptr_end = 0, len(children)
        prev_qmodule = None
        
        while (ptr < ptr_end):
            tmp_qoutput = qoutput if ptr == ptr_end-1 else True
            name, child_module = children[ptr][0], children[ptr][1]
            
            if type(child_module) in quant_modules:
                setattr(module, name, quant_modules[type(child_module)](child_module, config, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, config, qoutput=tmp_qoutput))
                prev_qmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_qmodule is not None:
                    prev_qmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, config, tmp_qoutput)
            ptr += 1
    
    # we replace all layers to be quantized with quantization-ready layers
    replace_module(model, config, qoutput=False)
    
    for name, module in model.named_modules():
        print(name, type(module))
    
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            print(name, module.out_mode)
    
    # we store all modules in the quantized model (weight_module or activation_module)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if 'weight' in name:
                w_list.append(module)
            elif 'act' in name:
                a_list.append(module)
    
    # set first and last layer to 8-bit
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)

    
    # set the last layer's output to 8-bit
    a_list[-1].set_bit(8)
    
    logger.info(f'Finished quantizing model: {str(model)}')
    
    return model


def get_calib_data(train_loader, num_samples):
    calib_data = []
    for batch in train_loader:
        calib_data.append(batch[0])
        if len(calib_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(calib_data, dim=0)[:num_samples]


def main(config_path):
    # get config for applying ptmq
    config = eval_utils.parse_config(config_path)
    eval_utils.set_seed(config.process.seed)
    
    train_loader, val_loader = eval_utils.load_data(**config.data)
    calib_data = get_calib_data(train_loader, config.quant.calibrate).cuda()
    
    model = load_model(config.model) # load original model
    search_fold_and_remove_bn(model) # remove+fold batchnorm layers
    
    # quanitze model if config.quant is defined
    if hasattr(config, 'quant'):
        model = quantize_model(model, config)
        
    model.cuda() # move model to GPU
    model.eval() # set model to evaluation mode
    
    fp_model = copy.deepcopy(model) # save copy of full precision model
    disable_all(fp_model) # disable all quantization
    
    """
    - fp 32 모델 weight 추출 
    
    for name, layer in fp_model.named_modules():
        # collect layer weight quantization params
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            # print(f"w_para from: {name}")
            #log_weight_distribution(layer.weight.data,'FP32')
""" 
    # set names for all ObserverBase modules
    # ObserverBase modules are used to store intermediate values during calibration
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)
    
    # calibration
    print("Starting model calibration...")
    with torch.no_grad():
        tik = time.time()
        enable_calib_without_quant(model, quantizer_type='act_fake_quant')
        model(calib_data[:256]).cuda()
        enable_calib_without_quant(model, quantizer_type='weight_fake_quant')
        model(calib_data[:2]).cuda()
        tok = time.time()
        logger.info(f"Calibration of {str(model)} took {tok - tik} seconds")
    print("Completed model calibration")
    
    print("Starting block reconstruction...")
    tik = time.time()
    # Block reconstruction (layer reconstruction for first & last layers)
    if hasattr(config.quant, 'recon'):
        enable_quantization(model)
        
        def recon_model(module, fp_module):
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info(f"Reconstructing module {str(child_module)}")
                    ptmq_reconstruction(model, fp_model, child_module, name, getattr(fp_module, name), calib_data, config.quant)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        
        recon_model(model, fp_model)
    tok = time.time()
    print("Completed block reconstruction")
    print(f"PTMQ block reconstruction took {tok - tik:.2f} seconds")
    
    
    a_qmodes = ["low", "med", "high"]
    w_qbit = config.quant.w_qconfig.bit
    a_qbits = [config.quant.a_qconfig_low.bit,
              config.quant.a_qconfig_med.bit,
              config.quant.a_qconfig_high.bit]
    
    enable_quantization(model)

    """
    for a_qmode, a_qbit in zip(a_qmodes, a_qbits):
        set_qmodel_block_aqbit(model, a_qmode)
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedBlock):
                print(name, module.out_mode)


                # QuantizedBlock의 출력 (f_l, f_m, f_h 등)을 가져와 히스토그램 계산
        print(f"Starting model evaluation of W{w_qbit}A{a_qbit} block reconstruction ({a_qmode})...")
        acc1, acc5 = eval_utils.validate_model(val_loader, model)
        
        print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")

          """
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default="/content/ptmq_log/config/gpu_config.yaml", type=str, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)