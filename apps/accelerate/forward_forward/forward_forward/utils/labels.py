from typing import List

import torch


class LabelsInjector:
    def __init__(self, labels: List):
        # save labels into a dict having label as key and a tensor of size
        #  len(labels) as value. The tensor contains ones up to the index of
        #  the label and zeros after.
        self.label_names = labels
        self.labels = [
            torch.nn.functional.one_hot(
                torch.tensor([i]), len(labels)
            ).reshape(-1)
            for i in range(len(labels))
        ]

    @torch.no_grad()
    def inject_train(self, input_image: torch.Tensor, labels: torch.Tensor):
        # inject label in the input image
        bs = input_image.shape[0]
        injecting_labels = torch.stack(
            [self.labels[label] for label in labels]
        )
        negative_injecting_labels = torch.stack(
            [
                self.labels[label]
                for label in select_random_different_label(
                    labels, len(self.labels)
                )
            ]
        )
        positive_images = torch.cat(
            [input_image.reshape(bs, -1), injecting_labels], dim=1
        )
        negative_images = torch.cat(
            [input_image.reshape(bs, -1), negative_injecting_labels], dim=1
        )
        images = torch.cat([positive_images, negative_images], dim=0)
        signs = torch.cat([torch.ones(bs), -torch.ones(bs)], dim=0)
        return images, signs

    @torch.no_grad()
    def inject_eval(self, input_image: torch.Tensor):
        # input image is expected to have batch size 1
        # TODO: FIX THIS BEHAVIOUR
        labels = torch.stack(self.labels).unsqueeze(0)
        labels = labels.repeat(input_image.shape[0], 1, 1)
        input_image = input_image.reshape(input_image.shape[0], -1).unsqueeze(
            1
        )
        replicated_input = input_image.repeat(1, len(self.labels), 1)
        new_input = torch.cat([replicated_input, labels], dim=2)
        return new_input  # .reshape(-1, new_input.shape[2])


def select_random_different_label(labels: torch.Tensor, n_classes: int):
    # select a random label different from the given one
    for label in enumerate(labels):
        samples = torch.randint(0, n_classes, (1,))
        while samples[0] == label:
            samples = torch.randint(0, n_classes, (1,))
        yield samples[0]
