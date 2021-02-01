import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg

class AbstractLearningRate(object):
    def __init__(self):
        pass

    def init_param(self):
        pass

    def init_param_jdata(self):
        pass

    def build(self):
        pass

    def start_lr(self):
        pass

    def value(self):
        pass

#TODO!
class LearningRateExp (AbstractLearningRate) :
    def __init__ (self, decay_steps, decay_rate, start_lr, stop_lr) : 
        self.decay_steps_ = decay_steps 
        self.decay_rate_  = decay_rate  
        self.start_lr_    = start_lr     
        self.stop_lr_     = stop_lr     

    @classmethod
    def init_param_jdata (self, jdata) :
        args = ClassArg()\
               .add('decay_steps',      int,    must = False)\
               .add('decay_rate',       float,  must = False)\
               .add('start_lr',         float,  must = True)\
               .add('stop_lr',          float,  must = False)

        cd           = args.parse(jdata)
        decay_steps_ = cd['decay_steps']
        decay_rate_  = cd['decay_rate']  
        start_lr_    = cd['start_lr']
        stop_lr_     = cd['stop_lr']     

        return self(decay_steps_, decay_rate_, start_lr_, stop_lr_)


    def build(self, global_step, stop_batch = None) :
        if stop_batch is None:            
            self.decay_steps_ = self.start_lr_    if self.start_lr_    is not None else 5000
            self.decay_rate_  = self.decay_rate_  if self.decay_rate_  is not None else 0.95
        else:
            self.stop_lr_     = self.stop_lr_     if self.stop_lr_     is not None else 5e-8
            default_ds = 100 if stop_batch // 10 > 100 else stop_batch // 100 + 1
            self.decay_steps_ = self.decay_steps_ if self.decay_steps_ is not None else default_ds
            if self.decay_steps_ >= stop_batch:
                self.decay_steps_ = default_ds
            self.decay_rate_ = np.exp(np.log(self.stop_lr_ / self.start_lr_) / (stop_batch / self.decay_steps_))
            
        return tf.train.exponential_decay(self.start_lr_, global_step, self.decay_steps_, self.decay_rate_, staircase=True)

    def start_lr(self) :
        return self.start_lr_

    def value (self, 
              batch) :
        return self.start_lr_ * np.power (self.decay_rate_, (batch // self.decay_steps_))

