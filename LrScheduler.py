import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class LrSchedulerWithWarmup(LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_total_steps, warmup_ratio, decay_ratio, last_epoch=-1):
        """
        自定义学习率调度器，包含预热阶段和衰减阶段。

        :param optimizer: 优化器对象
        :param num_warmup_steps: 预热阶段的步数
        :param num_total_steps: 总的训练步数
        :param warmup_ratio: 预热阶段之后的学习率衰减速度, 值越大衰减越快
        :param decay_ratio: 学习率的最大衰减比例
        :param last_epoch: 上一个训练周期的最后一个轮数，默认为-1
        """
        self.warmup_steps = num_warmup_steps
        self.total_steps = num_total_steps
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        super(LrSchedulerWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        计算并返回当前轮数的学习率。

        :return: 当前学习率
        """
        if self.last_epoch <= self.warmup_steps:
            # 预热阶段，学习率线性增加
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 衰减阶段，学习率按指数衰减
            return [
                base_lr * (1 - self.decay_ratio) * np.exp(-self.warmup_ratio * (self.last_epoch - self.warmup_steps))
                for
                base_lr in self.base_lrs]
