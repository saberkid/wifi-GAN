from torch.autograd import Variable

def mixup_batch(d, l, mixup=0.0):
    if mixup == 0:
        return d, l
    d2, l2 = one_batch()
    alpha = Variable(torch.randn(d1.size(0), 1).uniform_(0, mixup))
    if use_cuda:
        alpha = alpha.cuda()
    d = alpha * d1 + (1. - alpha) * d2
    l = alpha * l1 + (1. - alpha) * l2
    return d, l