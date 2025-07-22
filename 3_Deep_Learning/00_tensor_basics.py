import torch


def main():
    # Create tensors
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.rand(2, 2)
    print("Tensor a:\n", a)
    print("Tensor b:\n", b)

    # Basic operations
    print("a + b:\n", a + b)
    print("a @ b:\n", a @ b)

    # Autograd example
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("dy/dx at x=2:", x.grad.item())


if __name__ == "__main__":
    main()
