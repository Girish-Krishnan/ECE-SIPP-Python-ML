import torch


def main():
    """Showcase basic PyTorch tensor and autograd functionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create tensors on the selected device
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.rand(2, 2, device=device)
    print("Tensor a:\n", a)
    print("Tensor b:\n", b)

    # Basic operations
    print("a + b:\n", a + b)
    print("a @ b:\n", a @ b)

    # Autograd example
    x = torch.tensor([2.0], requires_grad=True, device=device)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("dy/dx at x=2:", x.grad.item())

    # Simple gradient descent for y = w*x + b
    w = torch.randn(1, requires_grad=True, device=device)
    b_param = torch.randn(1, requires_grad=True, device=device)
    opt = torch.optim.SGD([w, b_param], lr=0.1)
    loss_fn = torch.nn.MSELoss()

    xs = torch.linspace(-1, 1, 10, device=device)
    ys = 2 * xs + 1

    for _ in range(100):
        opt.zero_grad()
        preds = w * xs + b_param
        loss = loss_fn(preds, ys)
        loss.backward()
        opt.step()

    print("Fitted parameters:", w.item(), b_param.item())


if __name__ == "__main__":
    main()
