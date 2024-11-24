# -*- coding: utf-8 -*-
"""
@Time    : 11/12/2024 4:09 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import torch.nn as nn
import random
import numpy as np


class AvoidMaskNN(nn.Module):
    def __init__(self, num_segments=10):
        super(AvoidMaskNN, self).__init__()
        self.fc1 = nn.Linear(1 + num_segments, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, mask_encoding):
        x = torch.cat([x, mask_encoding], dim=1)  # Combine main input and mask encoding
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = torch.tanh(self.fc3(x))  # Ensure output stays within [-1, 1]
        return output


def generate_mask(num_segments=10):
    """
    Generates a random binary mask for the segments in [-1, 1].

    Parameters:
        num_segments (int): Number of segments (default 10).

    Returns:
        np.array: Binary mask of length `num_segments`.
    """
    # Randomly select some segments to be masked (1 represents masked)
    mask = np.zeros(num_segments)
    num_masked_segments = random.randint(1, num_segments-1)  # Random number of masked segments, never mask all segments
    masked_indices = random.sample(range(num_segments), num_masked_segments)  # randomly sample to choose which segment to mask
    mask[masked_indices] = 1
    return mask


def is_outside_masked_segments(output, mask_encoding, num_segments=10):
    """
    Checks if the output is outside all masked segments.

    Parameters:
        output (torch.Tensor): Model output value.
        mask_encoding (torch.Tensor): Binary mask encoding of segments.
        num_segments (int): Number of segments.

    Returns:
        bool: True if output is outside all masked segments, False otherwise.
    """
    segment_length = 2 / num_segments  # Length of each segment in [-1, 1]
    for i in range(num_segments):
        seg_min = -1 + i * segment_length
        seg_max = seg_min + segment_length
        if seg_min <= output < seg_max and mask_encoding[i] == 1:
            return False  # Output is inside a masked segment
    return True  # Output is outside all masked segments

# Parameters
num_segments = 10
num_eval_runs = 100  # Number of evaluations per evaluation episode
save_path = "./mask_model.pth"

# Load the model's parameters
model = AvoidMaskNN(num_segments=num_segments)  # Initialize a new model with the same architecture
model.load_state_dict(torch.load(save_path))
model.eval()  # Set to evaluation mode

# Evaluation Episode
success_count = 0
for _ in range(num_eval_runs):
    # Generate new random mask and input for each evaluation run
    mask_encoding = torch.tensor(generate_mask(num_segments)).float().unsqueeze(0)
    input_value = (torch.rand(1, 1) * 2 - 1).float()

    # Forward pass (evaluation mode, without gradient)
    with torch.no_grad():
        output = model(input_value, mask_encoding)

    # Check if output is outside masked segments
    if is_outside_masked_segments(output.item(), mask_encoding.squeeze()):
        success_count += 1

# Calculate and print success rate
success_rate = success_count / num_eval_runs * 100
print(f"Evaluation Episode - Success Rate: {success_rate}% (Out of {num_eval_runs} runs)")
