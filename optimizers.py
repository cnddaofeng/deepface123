import math
import random
import torch
import numpy as np
 
# 基础优化器抽象类
class Optimizer(object):
    def __init__(self, name=None):
        self.name = name
        self.weights = []
 
    def get_weights(self):
        return [w.cpu().numpy() for w in self.weights]
 
    def set_weights(self, weights):
        for i, w in enumerate(weights):
            self.weights[i].copy_(torch.from_numpy(w))
 
    def update(self, grads):
        raise NotImplementedError()
 
    # 兼容 PyTorch 标准接口
    def zero_grad(self):
        pass
 
    def step(self, closure=None):
        raise NotImplementedError()
 
class SophiaG(Optimizer):
    """
    Sophia-G: Second-order Clipped Optimizer
    这是一个独立的类，不依赖 core.leras.nn
    """
    def __init__(self, params, lr=2e-4, beta1=0.9, beta2=0.99, rho=0.04, 
                 weight_decay=0.1, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None):
        super().__init__(name=name)
        
        if isinstance(params, (list, tuple)) and len(params) > 0:
            self.tensors = params
        else:
            self.tensors = list(params)
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.weight_decay = weight_decay
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm
        self.initialized = False
        self._step = 0
 
    def initialize_variables(self):
        if self.initialized:
            return
        
        device = self.tensors[0].device if len(self.tensors) > 0 else torch.device('cpu')
        
        self.weights = [
            torch.tensor([0], dtype=torch.int64, device=device),  # step
            *[torch.zeros_like(t) for t in self.tensors],  # 一阶动量
            *[torch.zeros_like(t) for t in self.tensors]   # 二阶 Hessian 估计
        ]
        self.initialized = True
 
    @torch.no_grad()
    def step(self, closure=None):
        """标准 PyTorch 优化器接口"""
        if not self.initialized:
            self.initialize_variables()
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        grads = []
        for p in self.tensors:
            if p.grad is not None:
                grads.append(p.grad.data)
            else:
                grads.append(None)
        
        self._update_with_grads(grads)
        return loss
 
    @torch.no_grad()
    def update(self, grads):
        """兼容旧版接口"""
        if not self.initialized:
            self.initialize_variables()
        self._update_with_grads(grads)
 
    def _update_with_grads(self, grads):
        """内部更新逻辑"""
        n_params = len(self.tensors)
        iter_var = self.weights[0]
        m_vars = self.weights[1 : n_params + 1]
        h_vars = self.weights[n_params + 1:]
        
        iter_var += 1
        self._step = iter_var.item()
        lr = self.lr
 
        # 学习率调度
        if self.lr_dropout < 1.0 and random.random() > self.lr_dropout:
            return
        if self.lr_cos > 0:
            t = (self._step - 1) % self.lr_cos
            lr = lr * (1.0 + math.cos(math.pi * t / self.lr_cos)) / 2.0
 
        # 梯度裁剪
        if self.clipnorm > 0.0:
            total_norm = torch.sqrt(sum(g.norm()**2 for g in grads if g is not None))
            if total_norm > self.clipnorm:
                clip_coef = self.clipnorm / (total_norm + 1e-6)
                grads = [g * clip_coef if g is not None else None for g in grads]
 
        # Sophia-G 核心更新
        for i, (p, g) in enumerate(zip(self.tensors, grads)):
            if g is None:
                continue
            
            m, h = m_vars[i], h_vars[i]
            
            # 更新一阶动量
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            
            # 更新 Hessian 估计
            h.mul_(self.beta2).add_(g.abs(), alpha=1.0 - self.beta2)
 
            # 二阶裁剪更新
            denom = torch.clamp(h, min=self.rho)
            update = m / denom
            
            # 权重衰减
            if self.weight_decay > 0:
                p.add_(p.data, alpha=-lr * self.weight_decay)
            
            # 参数更新
            p.add_(update, alpha=-lr)