# -*- coding: utf-8 -*-
"""
@Time    : 11/20/2024 12:20 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

# -*- coding: utf-8 -*-
"""
@Time    : 11/12/2024 10:47 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention


class AvoidMaskNN(nn.Module):
    def __init__(self, num_segments=10):
        super(AvoidMaskNN, self).__init__()
        self.fc1 = nn.Linear(1 + num_segments, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 1)
        self.fc3 = nn.Sequential(nn.Linear(128, 1), nn.Tanh())
        # self.fc3 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, mask_encoding):
        x = torch.cat([x, mask_encoding], dim=1)  # Combine main input and mask encoding
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)  # Ensure output stays within [-1, 1]
        return output


class AvoidMaskNN_v2(nn.Module):
    def __init__(self, num_segments=10):
        super(AvoidMaskNN_v2, self).__init__()
        self.fc1 = nn.Linear(1 + num_segments, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, x, mask_encoding):
        x = torch.cat([x, mask_encoding], dim=1)  # Combine main input and mask encoding
        # x = torch.relu(self.fc1(x))
        x = torch.relu(self.layer_norm(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        output = torch.tanh(self.fc5(x))  # Ensure output stays within [-1, 1]
        return output


class AvoidMaskNN_with_att(nn.Module):
    def __init__(self, num_segments=10):
        super(AvoidMaskNN_with_att, self).__init__()
        self.q_projection = nn.Linear(1+num_segments, 128)
        self.k_projection = nn.Linear(num_segments+1, 128)
        self.v_projection = nn.Linear(1+num_segments, 128)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=4)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, mask_encoding):
        combine_x = torch.cat([x, mask_encoding], dim=2)  # Combine main input and mask encoding
        x_q = self.q_projection(combine_x)
        x_k = self.k_projection(combine_x)
        x_v = self.v_projection(combine_x)
        att_out = self.ScaledDotProductAttention(x_q, x_k, x_v)
        x = torch.relu(self.fc3(att_out))
        output = torch.tanh(self.fc4(x))  # Ensure output stays within [-1, 1]
        return output


def custom_loss(output, mask_encoding, segments):
    """
    Custom loss that penalizes outputs within masked segments based on their distance
    from the nearest non-masked segment.

    Parameters:
        output (torch.Tensor): Model output values of shape (batch_size, 1).
        mask_encoding (torch.Tensor): Binary mask encoding of segments of shape (num_segments,).
        segments (list of tuples): Predefined list of segment boundaries as (seg_min, seg_max) tuples.

    Returns:
        torch.Tensor: Mean loss value across the batch.
    """
    # Convert segments into tensors
    seg_mins = torch.tensor([seg[0] for seg in segments], dtype=output.dtype, device=output.device)
    seg_maxs = torch.tensor([seg[1] for seg in segments], dtype=output.dtype, device=output.device)

    # Ensure mask_encoding is on the same device as output
    mask_encoding = mask_encoding.to(output.device)

    # Expand output to compare with all segments simultaneously
    output_expanded = output.view(-1, 1)  # Shape: (batch_size, 1)

    # Identify masked segments
    masked_seg_mins = seg_mins[mask_encoding == 1]
    masked_seg_maxs = seg_maxs[mask_encoding == 1]
    unmasked_seg_mins = seg_mins[mask_encoding == 0]
    unmasked_seg_maxs = seg_maxs[mask_encoding == 0]

    # Initialize penalties as zeros
    penalties = torch.zeros_like(output, requires_grad=True)

    for i in range(len(output)):
        out_val = output[i]

        # Check if output is within any masked segment
        in_masked_segment = ((out_val >= masked_seg_mins) & (out_val < masked_seg_maxs)).any()

        if in_masked_segment:
            # Compute distances to all unmasked segment boundaries
            distances_to_unmasked = torch.cat([
                torch.abs(out_val - unmasked_seg_mins),
                torch.abs(out_val - unmasked_seg_maxs)
            ])

            # Find the minimum distance to a non-masked boundary
            min_distance = distances_to_unmasked.min()

            # Calculate penalty as an inverse function of the distance (closer to 1 if far away, closer to 0 if near)
            penalties = penalties + (1 - torch.exp(-min_distance))  # Penalty approaches 1 as distance decreases

    # Return the mean penalty across the batch
    return penalties.mean()


def custom_loss_v2(output, mask_encoding, segments, input_value):
    """
    Custom loss that penalizes outputs within masked segments based on their distance
    from the nearest non-masked segment.

    Parameters:
        output (torch.Tensor): Model output values of shape (batch_size, 1).
        mask_encoding (torch.Tensor): Binary mask encoding of segments of shape (num_segments,).
        segments (list of tuples): Predefined list of segment boundaries as (seg_min, seg_max) tuples.

    Returns:
        torch.Tensor: Mean loss value across the batch.
    """

    seg_mins = torch.tensor(
        [[seg[0] for seg in batch_segments] for batch_segments in segments])  # Shape: (batch_size, num_segments)
    seg_maxs = torch.tensor([[seg[1] for seg in batch_segments] for batch_segments in segments])

    # Ensure mask_encoding is on the same device as output
    mask_encoding = mask_encoding.to(output.device)

    masked_seg_mins = torch.where(mask_encoding == 1, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    masked_seg_maxs = torch.where(mask_encoding == 1, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))
    unmasked_seg_mins = torch.where(mask_encoding == 0, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    unmasked_seg_maxs = torch.where(mask_encoding == 0, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))

    # Condition 1: Input is outside all masked segments
    in_masked_segment = ((input_value >= masked_seg_mins) & (input_value < masked_seg_maxs)).any(dim=1, keepdim=True)
    outside_masked_segments = ~in_masked_segment

    # Condition 2: All segments are masked
    all_segments_masked = mask_encoding.sum(dim=1, keepdim=True) == mask_encoding.size(1)

    # Condition 3: Input is inside a masked segment but not all segments are masked
    inside_masked_segments = in_masked_segment & ~all_segments_masked

    # Initialize penalties
    penalties = torch.zeros_like(output)

    # Penalties for Condition 1
    penalties[outside_masked_segments] = penalties[outside_masked_segments] + (
            output[outside_masked_segments] - input_value[outside_masked_segments]).pow(2)

    # Penalties for Condition 2
    penalties[all_segments_masked] = penalties[all_segments_masked] + (
            output[all_segments_masked] - output.mean()).pow(2)

    # Penalties for Condition 3
    if inside_masked_segments.any():
        # Select relevant outputs
        out_val = output[inside_masked_segments]

        # Select unmasked segment boundaries for relevant outputs
        filtered_unmasked_seg_mins = unmasked_seg_mins[inside_masked_segments.squeeze(), :]
        filtered_unmasked_seg_maxs = unmasked_seg_maxs[inside_masked_segments.squeeze(), :]

        # Expand `out_val` to align with segment boundaries
        out_val_expanded = out_val.unsqueeze(1)

        # Compute distances to nearest unmasked segments
        distances_to_unmasked = torch.cat([
            torch.abs(out_val_expanded - filtered_unmasked_seg_mins),
            torch.abs(out_val_expanded - filtered_unmasked_seg_maxs)
        ], dim=1)
        min_distance = distances_to_unmasked.min(dim=1, keepdim=True).values

        # Scatter min_distance back to the full batch shape
        full_min_distance = torch.zeros_like(output)  # Shape: (32, 1)
        full_min_distance[inside_masked_segments] = min_distance.squeeze(1)   # Align with `inside_masked_segments`

        # Apply penalties for Condition 3
        penalties = penalties + (1 - torch.exp(-full_min_distance))
    return penalties.mean()


def custom_loss_all_mask(output, mask_encoding, segments, input_value):
    """
    Custom loss that penalizes outputs within masked segments based on their distance
    from the nearest non-masked segment.

    Parameters:
        output (torch.Tensor): Model output values of shape (batch_size, 1).
        mask_encoding (torch.Tensor): Binary mask encoding of segments of shape (num_segments,).
        segments (list of tuples): Predefined list of segment boundaries as (seg_min, seg_max) tuples.

    Returns:
        torch.Tensor: Mean loss value across the batch.
    """

    seg_mins = torch.tensor(
        [[seg[0] for seg in batch_segments] for batch_segments in segments])  # Shape: (batch_size, num_segments)
    seg_maxs = torch.tensor([[seg[1] for seg in batch_segments] for batch_segments in segments])

    # Ensure mask_encoding is on the same device as output
    mask_encoding = mask_encoding.to(output.device)

    masked_seg_mins = torch.where(mask_encoding == 1, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    masked_seg_maxs = torch.where(mask_encoding == 1, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))
    unmasked_seg_mins = torch.where(mask_encoding == 0, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    unmasked_seg_maxs = torch.where(mask_encoding == 0, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))

    # Condition 2: All segments are masked
    all_segments_masked = mask_encoding.sum(dim=1, keepdim=True) == mask_encoding.size(1)

    # Initialize penalties
    penalties = torch.zeros_like(output)

    # Penalties for Condition 2
    penalties[all_segments_masked] = penalties[all_segments_masked] + (output[all_segments_masked] -
                                                                       input_value[all_segments_masked]).pow(2)

    return penalties.mean()


def custom_loss_in_mask(output, mask_encoding, segments, input_value):
    seg_mins = torch.tensor(
        [[seg[0] for seg in batch_segments] for batch_segments in segments])  # Shape: (batch_size, num_segments)
    seg_maxs = torch.tensor([[seg[1] for seg in batch_segments] for batch_segments in segments])

    # Ensure mask_encoding is on the same device as output
    mask_encoding = mask_encoding.to(output.device)
    large_constant = 1e6
    masked_seg_mins = torch.where(mask_encoding == 1, seg_mins, torch.tensor(large_constant, device=seg_mins.device))
    masked_seg_maxs = torch.where(mask_encoding == 1, seg_maxs, torch.tensor(large_constant, device=seg_maxs.device))
    unmasked_seg_mins = torch.where(mask_encoding == 0, seg_mins, torch.tensor(large_constant, device=seg_mins.device))
    unmasked_seg_maxs = torch.where(mask_encoding == 0, seg_maxs, torch.tensor(large_constant, device=seg_maxs.device))

    in_masked_segment = ((input_value >= masked_seg_mins) & (input_value < masked_seg_maxs)).any(dim=1, keepdim=True)
    all_segments_masked = mask_encoding.sum(dim=1, keepdim=True) == mask_encoding.size(1)

    # Condition 3: Input is inside a masked segment but not all segments are masked
    inside_masked_segments = in_masked_segment & ~all_segments_masked

    # Initialize penalties
    penalties = torch.zeros_like(output)

    # Penalties for Condition 3
    if inside_masked_segments.any():
        # Select relevant outputs
        out_val = output[inside_masked_segments]

        # Select unmasked segment boundaries for relevant outputs
        filtered_unmasked_seg_mins = unmasked_seg_mins[inside_masked_segments.squeeze(), :]
        filtered_unmasked_seg_maxs = unmasked_seg_maxs[inside_masked_segments.squeeze(), :]

        # Expand `out_val` to align with segment boundaries
        out_val_expanded = out_val.unsqueeze(1)

        input_value_in_mask = input_value[inside_masked_segments].unsqueeze(1)
        distances_to_unmasked_v2 = torch.cat([
            torch.abs(input_value_in_mask - filtered_unmasked_seg_mins),
            torch.abs(input_value_in_mask - filtered_unmasked_seg_maxs)
        ], dim=1)

        dist_to_nearest_unmask_seg = distances_to_unmasked_v2.min(dim=1, keepdim=True)  # minimum index, for cat[filtered_unmasked_seg_mins, filtered_unmasked_seg_maxs]
        combine_filtered_unmasked_seg_min_max = torch.cat([filtered_unmasked_seg_mins, filtered_unmasked_seg_maxs],dim=1)

        first_portion_nearest_seg = combine_filtered_unmasked_seg_min_max.gather(1, dist_to_nearest_unmask_seg.indices)

        final_portion_nearest_seg = first_portion_nearest_seg + torch.where(dist_to_nearest_unmask_seg.indices <= 9, 0.2, -0.2)

        sum_nearest_seg = torch.cat((first_portion_nearest_seg, final_portion_nearest_seg), dim=1)

        middle_nearest_seg = sum_nearest_seg.mean(dim=1, keepdim=True)

        # Scatter min_distance back to the full batch shape
        full_min_distance = torch.zeros_like(output)  # Shape: (32, 1)
        full_min_distance[inside_masked_segments] = (out_val_expanded - middle_nearest_seg).squeeze(1)   # Align with `inside_masked_segments`
        # full_min_distance[inside_masked_segments] = (out_val_expanded - first_portion_nearest_seg).squeeze(1)   # Align with `inside_masked_segments`


        # Apply penalties for Condition 3
        # penalties = penalties + (1 - torch.exp(-full_min_distance))
        penalties = penalties + full_min_distance.pow(2)
    return penalties.mean()


def custom_loss_out_mask(output, mask_encoding, segments, input_value):

    seg_mins = torch.tensor(
        [[seg[0] for seg in batch_segments] for batch_segments in segments])  # Shape: (batch_size, num_segments)
    seg_maxs = torch.tensor([[seg[1] for seg in batch_segments] for batch_segments in segments])

    # Ensure mask_encoding is on the same device as output
    mask_encoding = mask_encoding.to(output.device)

    masked_seg_mins = torch.where(mask_encoding == 1, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    masked_seg_maxs = torch.where(mask_encoding == 1, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))
    unmasked_seg_mins = torch.where(mask_encoding == 0, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
    unmasked_seg_maxs = torch.where(mask_encoding == 0, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))

    # Condition 1: Input is outside all masked segments
    in_masked_segment = ((input_value >= masked_seg_mins) & (input_value < masked_seg_maxs)).any(dim=1, keepdim=True)
    outside_masked_segments = ~in_masked_segment

    # Initialize penalties
    penalties = torch.zeros_like(output)

    # Penalties for Condition 1
    penalties[outside_masked_segments] = penalties[outside_masked_segments] + (
            output[outside_masked_segments] - input_value[outside_masked_segments]).pow(2)

    return penalties.mean()


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
    # num_masked_segments = random.randint(1, num_segments)  # Random number of masked segments, possible to mask all
    num_masked_segments = random.randint(1, num_segments-1)  # Random number of masked segments, not possible to mask all
    # num_masked_segments = num_segments  # mask all
    masked_indices = random.sample(range(num_segments), num_masked_segments)  # randomly sample to choose which segment to mask
    mask[masked_indices] = 1
    return mask


def is_outside_masked_segments(output, mask_encoding, segments):
    """
    Check if output is outside all masked segments.

    Parameters:
        output (float): The output value to check.
        mask_encoding (torch.Tensor): Binary mask encoding of segments.
        segments (list of tuples): Segment boundaries as (seg_min, seg_max).

    Returns:
        bool: True if output is outside all masked segments, False otherwise.
    """
    for i, (seg_min, seg_max) in enumerate(segments):
        if seg_min <= output < seg_max and mask_encoding[i] == 1:
            return False  # Output is inside a masked segment
    return True  # Output is outside all masked segments


def is_inside_masked_segments(input_value, mask_encoding, segments):
    """
    Check if input_value is inside any masked segment.

    Parameters:
        input_value (float): The input value to check.
        mask_encoding (torch.Tensor): Binary mask encoding of segments.
        segments (list of tuples): Segment boundaries as (seg_min, seg_max).

    Returns:
        bool: True if input_value is inside a masked segment, False otherwise.
    """
    segment_length = 2 / len(segments)  # Length of each segment in [-1, 1]
    for i, (seg_min, seg_max) in enumerate(segments):
        if seg_min <= input_value < seg_max and mask_encoding[i] == 1:
            return True  # Input is inside a masked segment
    return False  # Input is outside all masked segments

# Q=torch.randn(50,1,10)
# K=torch.randn(50,1,10)
# V=torch.randn(50,1,10)
#
# sa = ScaledDotProductAttention(d_model=10, d_k=10, d_v=10, h=1)
# output=sa(Q,K,V)  # Q, K, V
# print(output.shape)


# Parameters
num_segments = 10
# epochs = 200000
epochs = 120000
learning_rate = 0.001
eval_interval = 100  # Perform evaluation every 100 epochs
num_eval_runs = 50  # Number of evaluations per evaluation episode
threshold = 0.01
# Define batch size
batch_size = 32  # Number of input values per batch

# ----- other choice button ----
# use_attention = True
use_attention = False

# Initialize a list to store success rates for every 100 episodes
success_rates = []

# Define the save path
save_path = "./mask_model_out_mask_wTanh_v2.pth"

# Define the start and end of the interval
start, end = -1, 1

# Calculate the segment length
segment_length = (end - start) / num_segments

# Generate segment boundaries
# boundaries = np.linspace(start, end, num_segments + 1)

# Generate segments as tuples
# segments = [(boundaries[i], boundaries[i+1]) for i in range(num_segments)]

# Generate segment boundaries for each batch
segment_boundaries = torch.linspace(start, end, num_segments + 1).unsqueeze(0).expand(batch_size, -1)
segments = [
    [(segment_boundaries[b, i].item(), segment_boundaries[b, i + 1].item()) for i in range(num_segments)]
    for b in range(batch_size)
]

mask_out = generate_mask(len(segments))  # 1x10 binary encoder

# Initialize model and optimizer
if use_attention:
    model = AvoidMaskNN_with_att(num_segments=num_segments)
else:
    model = AvoidMaskNN(num_segments=num_segments)
    # model = AvoidMaskNN_v2(num_segments=num_segments)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):

    # Generate a random mask
    # mask_encoding = torch.tensor(generate_mask(num_segments)).float().unsqueeze(0).requires_grad_()  # Add batch dimension
    mask_encodings = torch.stack([torch.tensor(generate_mask(num_segments)).float() for _ in range(batch_size)])
    if use_attention:
        mask_encodings = mask_encodings.unsqueeze(1)  # ensure we add sequence length for the attention module
    # Generate random input value in range [-1, 1]
    # input_value = (torch.rand(1, 1) * 2 - 1).clone().detach().requires_grad_()
    input_values = (torch.rand(batch_size, 1) * 2 - 1).float().requires_grad_()
    if use_attention:
        input_values = input_values.unsqueeze(1)  # ensure we add sequence length for the attention module
    # Forward pass
    outputs = model(input_values, mask_encodings)  # here no attention, both are (b_s, dim), when attention present we need to add seq_length

    # Calculate loss
    # loss = custom_loss(output, mask_encoding.squeeze(), segments)
    if use_attention:
        loss = custom_loss_v2(outputs.squeeze(1), mask_encodings.squeeze(1), segments, input_values.squeeze(
            1))  # here no or with attention outputs and input_values shape is (b_s, dim), mask shape is (b_s, dim)
    else:
        # loss = custom_loss_v2(outputs, mask_encodings, segments, input_values)  # here no or with attention outputs shape is (b_s, dim), mask shape is (b_s, dim)
        # loss = custom_loss_all_mask(outputs, mask_encodings, segments, input_values)  # here no or with attention outputs shape is (b_s, dim), mask shape is (b_s, dim)
        loss = custom_loss_out_mask(outputs, mask_encodings, segments, input_values)  # here no or with attention outputs shape is (b_s, dim), mask shape is (b_s, dim)
        # loss = custom_loss_in_mask(outputs, mask_encodings, segments, input_values)  # here no or with attention outputs shape is (b_s, dim), mask shape is (b_s, dim)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 100 epochs and run evaluation
    if (epoch + 1) % eval_interval == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Evaluation Episode
        success_count = 0
        invalid_count_per_eval_run = 0
        for _ in range(num_eval_runs):

            # Generate new random mask and input for each evaluation run
            # mask_encoding = torch.tensor(generate_mask(num_segments)).float().unsqueeze(0)
            mask_encodings = torch.stack([torch.tensor(generate_mask(num_segments)).float() for _ in range(batch_size)])

            # input_value = (torch.rand(1, 1) * 2 - 1).float()
            input_values = (torch.rand(batch_size, 1) * 2 - 1).float()

            # Forward pass (evaluation mode, without gradient)
            with torch.no_grad():
                if use_attention:
                    outputs = model(input_values.unsqueeze(1), mask_encodings.unsqueeze(1))  # with attention
                else:
                    outputs = model(input_values, mask_encodings)  # no attention


            # input_val = input_values.item()
            input_val = input_values
            # output_val = outputs.item()

            if use_attention:
                output_val = outputs.squeeze(1)  # for attention
            else:
                output_val = outputs

            # among outputs there will be three mutually exclusive conditions
            seg_mins = torch.tensor(
                [[seg[0] for seg in batch_segments] for batch_segments in
                 segments])  # Shape: (batch_size, num_segments)
            seg_maxs = torch.tensor([[seg[1] for seg in batch_segments] for batch_segments in segments])

            # Ensure mask_encoding is on the same device as output
            mask_encoding = mask_encodings.to(output_val.device)

            masked_seg_mins = torch.where(mask_encoding == 1, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
            masked_seg_maxs = torch.where(mask_encoding == 1, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))
            unmasked_seg_mins = torch.where(mask_encoding == 0, seg_mins, torch.tensor(float('inf'), device=seg_mins.device))
            unmasked_seg_maxs = torch.where(mask_encoding == 0, seg_maxs, torch.tensor(float('inf'), device=seg_maxs.device))

            # Condition 1: Input is outside all masked segments, output should be same as input
            in_masked_segment = ((input_val >= masked_seg_mins) & (input_val < masked_seg_maxs)).any(dim=1,
                                                                                                       keepdim=True)
            outside_masked_segments = ~in_masked_segment
            difference_outside_mask = output_val[outside_masked_segments] - input_val[outside_masked_segments]
            # Find values below the threshold
            below_threshold = difference_outside_mask < threshold
            # Count how many results are below the threshold
            count_below_threshold_outside_mask = torch.sum(below_threshold).item()

            success_count = success_count + count_below_threshold_outside_mask  # condition 1 success_count

            # count how many binary mask condition is out of the range for the current NN
            invalid_count_per_eval_run = invalid_count_per_eval_run + torch.sum(outside_masked_segments == False).item()

            # condition 2: All segments are masked, output should be same as input
            all_segments_masked = mask_encoding.sum(dim=1, keepdim=True) == mask_encoding.size(1)
            difference_all_seg_mask = output_val[all_segments_masked] - input_val[all_segments_masked]
            below_threshold_all_mask = difference_all_seg_mask < threshold
            count_below_threshold_all_seg_mask = torch.sum(below_threshold_all_mask).item()
            # if count_below_threshold_all_seg_mask > 0:
            #     print("condition 2 available")

            # success_count = success_count + count_below_threshold_all_seg_mask  # condition 2 success count

            # condition 3: Input is inside a masked segment, and output must be within the nearest unmasked segments
            inside_masked_segments = in_masked_segment & ~all_segments_masked  # the specific rows that input falls into the masked segments
            # Select relevant inputs and outputs
            input_vals = input_val[inside_masked_segments]  # Shape: (num_selected,)
            output_vals = output_val[inside_masked_segments]  # Shape: (num_selected,)

            # Select unmasked segment boundaries for relevant inputs

            # Expand the mask to align with unmasked_seg_mins (if necessary)
            inside_masked_segments_expanded = (inside_masked_segments.squeeze(-1).unsqueeze(1).
                                               expand(-1, unmasked_seg_mins.size(1)))  # Shape: (batch_size, num_segments)

            # Filter unmasked segment boundaries using the expanded mask
            filtered_unmasked_seg_mins = torch.masked_select(unmasked_seg_mins, inside_masked_segments_expanded).view(
                -1, unmasked_seg_mins.size(1))
            filtered_unmasked_seg_maxs = torch.masked_select(unmasked_seg_maxs, inside_masked_segments_expanded).view(
                -1, unmasked_seg_maxs.size(1))

            # Expand outputs to align with unmasked segment boundaries
            output_vals_expanded = output_vals.unsqueeze(1)  # Shape: (num_selected, 1)
            # Calculate distances to unmasked segments
            distances_to_unmasked_mins = torch.abs(
                output_vals_expanded - filtered_unmasked_seg_mins)  # Shape: (num_selected, num_segments)
            distances_to_unmasked_maxs = torch.abs(
                output_vals_expanded - filtered_unmasked_seg_maxs)  # Shape: (num_selected, num_segments)

            input_value_in_mask = input_vals.unsqueeze(1)
            distances_to_unmasked_v2 = torch.cat([
                torch.abs(input_value_in_mask - filtered_unmasked_seg_mins),
                torch.abs(input_value_in_mask - filtered_unmasked_seg_maxs)
            ], dim=1)

            dist_to_nearest_unmask_seg = distances_to_unmasked_v2.min(dim=1,
                                                                      keepdim=True)  # minimum index, for cat[filtered_unmasked_seg_mins, filtered_unmasked_seg_maxs]
            combine_filtered_unmasked_seg_min_max = torch.cat([filtered_unmasked_seg_mins, filtered_unmasked_seg_maxs],
                                                              dim=1)

            first_portion_nearest_seg = combine_filtered_unmasked_seg_min_max.gather(1,
                                                                                     dist_to_nearest_unmask_seg.indices)

            final_portion_nearest_seg = first_portion_nearest_seg + torch.where(dist_to_nearest_unmask_seg.indices <= 9,
                                                                                0.2, -0.2)

            nearest_unmasked_segment = torch.cat([first_portion_nearest_seg, final_portion_nearest_seg], dim=1)

            nearest_unmasked_segment_sorted, _ = nearest_unmasked_segment.sort(dim=1)  # ensure is from  min to max

            # Extract the min and max columns
            nearest_unmasked_segment_sorted_min = nearest_unmasked_segment_sorted[:, 0]
            nearest_unmasked_segment_sorted_max = nearest_unmasked_segment_sorted[:, 1]

            within_range = (output_vals >= nearest_unmasked_segment_sorted_min) & (output_vals <= nearest_unmasked_segment_sorted_max)

            count_true_output_within_nearest = within_range.sum().item()
            count_false_output_within_nearest = within_range.numel() - count_true_output_within_nearest

            # success_count = success_count + count_true_output_within_nearest  # condition 3 success count
            #
            # invalid_count_per_eval_run = invalid_count_per_eval_run + torch.sum(inside_masked_segments == False).item()

        # Calculate and print success rate
        success_rate_for_one_eps = success_count / ((batch_size*num_eval_runs)-invalid_count_per_eval_run)
        success_rate_for_one_eps_percent = success_rate_for_one_eps * 100
        success_rates.append(success_rate_for_one_eps_percent)
        # print(f"Evaluation Episode - Success Rate: {success_rate}% (Out of {num_eval_runs} runs)")

        # Print average success rate every 100 episodes
        if len(success_rates) >= eval_interval:
            avg_success_rate = sum(success_rates[-eval_interval:]) / eval_interval
            print(f"Average Success Rate (Last {eval_interval} Episodes): {avg_success_rate:.2f}%")

torch.save(model.state_dict(), save_path)
print(f"Final model parameters saved at epoch {epoch + 1}")