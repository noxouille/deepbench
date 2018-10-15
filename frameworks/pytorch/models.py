from time import time

import torch
# this turns on auto tuner which optimizes performance
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print('cuda version=', torch.version.cuda)
    print('cudnn version=', torch.backends.cudnn.version())

class pytorch_base:

    def __init__(self, model_name, precision, image_shape, batch_size):
        if torch.cuda.is_available():
        	self.model = getattr(torchvision.models, model_name)().to(device) if precision == 'fp32' \
				else getattr(torchvision.models, model_name)().to(device).half()
        	x = torch.rand(batch_size, 3, image_shape[0], image_shape[1])
        	self.eval_input = torch.autograd.Variable(x).to(device) if precision == 'fp32' \
				else torch.autograd.Variable(x).to(device).half()
        	self.train_input = torch.autograd.Variable(x, requires_grad=True).to(device) if precision == 'fp32' \
				else torch.autograd.Variable(x, requires_grad=True).to(device).half()
        else:
            self.model = getattr(torchvision.models, model_name)()
            x = torch.rand(batch_size, 3, image_shape[0], image_shape[1])
            self.eval_input = torch.autograd.Variable(x)
            self.train_input = torch.autograd.Variable(x, requires_grad=True)

    def eval(self, num_iterations, num_warmups):
        self.model.eval()
        durations = []
        for i in range(num_iterations + num_warmups):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time()
            self.model(self.eval_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations

    def train(self, num_iterations, num_warmups):
        self.model.train()
        durations = []
        for i in range(num_iterations + num_warmups):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time()
            self.model.zero_grad()
            out = self.model(self.train_input)
            loss = out.sum()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations

class vgg16(pytorch_base):

  def __init__(self, precision, image_shape, batch_size):
    super().__init__('vgg16', precision, image_shape, batch_size)


class resnet152(pytorch_base):

  def __init__(self, precision, image_shape, batch_size):
    super().__init__('resnet152', precision, image_shape, batch_size)


class densenet161(pytorch_base):

  def __init__(self, precision, image_shape, batch_size):
    super().__init__('densenet161', precision, image_shape, batch_size)

