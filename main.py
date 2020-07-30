from torch_nn import *

# 0 . Structure (LeNet)
net = Net()
#print(net)

# 1. the learnable param of a model are returned by net.parameters()
params = list(net.parameters())
#print(len(params))
#print(params[0].size()) #conv1's .weight

# try a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
#print(out)

net.zero_grad()
out.backward(torch.randn((1, 10)))

output = net(input)
target = torch.randn(10) # which is dummy target
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)