import torch
import torch.nn as nn
import torch.optim as optim


def main():
    distances = torch.tensor([[1.1], [2], [2.8], [4.5]], dtype=torch.float32)
    times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)
    inputs = distances

    model = nn.Sequential(nn.Linear(1, 1))

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for _ in range(500000):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, times)
        loss.backward()
        optimizer.step()

    w = model[0].weight.item()
    b = model[0].bias.item()
    print(f"Learned: time ≈ {w:.5f} * distance + {b:.5f}")

    with torch.no_grad():
        test_distance = torch.tensor([[3]], dtype=torch.float32)
        predicted_time = model(test_distance)
        print(f"Predicted time: {predicted_time.item():.5f} minutes")


if __name__ == "__main__":
    main()
