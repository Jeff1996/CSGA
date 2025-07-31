from mmengine.hooks import Hook
import torch
from mmdet.registry import HOOKS


@HOOKS.register_module()
class EmptyCacheHook(Hook):
    """在验证前清空 CUDA 缓存的 Hook"""
    def after_train(self, runner):
        torch.cuda.empty_cache()
        runner.logger.info('Cleared CUDA cache before validation')
        