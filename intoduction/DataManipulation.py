import torch




def main():
    x = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(3, -1)
    xt = torch.randn(size=(4, 3))
    # print(x)
    y = x@xt
    print(y)
    print(y.shape)
    y.backward(gradient=torch.ones(3))
    print(x.grad)
    
    
    # print(x.grad)
if __name__ == '__main__':
    main()