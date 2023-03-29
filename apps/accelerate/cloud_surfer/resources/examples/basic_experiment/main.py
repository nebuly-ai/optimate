import torch
import torchvision.models as models

from speedster.api.functions import optimize_model


def main():
    data = [((torch.randn(1, 3, 256, 256),), 0) for _ in range(100)]
    model = models.resnet50()
    m = optimize_model(
        model=model,
        input_data=data,
    )


if __name__ == "__main__":
    main()
