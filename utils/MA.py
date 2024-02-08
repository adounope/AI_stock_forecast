import torch

def MA(data, days): #return moving average tensor
    data = data.unfold(0, days, 1)
    return data.mean(dim = 1)





































#since moving average is just convolution with averaging kernel
#maybe we should let the AI to learn a kernel to form MA lines?