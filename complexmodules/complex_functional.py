from locale import normalize
from math import ceil
import torch
import torch.nn.functional as F

import numpy as np

Tensor = torch.Tensor


def max_poolnd(
    pool, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):

    # Use torch.maxpool to get indices.
    _, indices = pool(
        abs(input),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        return_indices=True,
    )

    flat = input.flatten(2, -1)
    ix = indices.flatten(2, -1)
    temp = torch.gather(flat, -1, ix)
    out = temp.reshape(input.shape[:2] + indices.shape[2:])

    return out


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """Complex 2d max pooling on the complex tensor `B x c_in x H x W`."""

    return max_poolnd(
        F.max_pool2d, input, kernel_size, stride, padding, dilation, ceil_mode
    )


def complex_avg_pool2d(
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    a = F.avg_pool2d(
        input.real,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    b = F.avg_pool2d(
        input.imag,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )

    z = torch.complex(a, b)

    return z


def complex_adaptive_avg_pool2d(input, output_size):

    a = F.adaptive_avg_pool2d(input.real, output_size)
    b = F.adaptive_avg_pool2d(input.imag, output_size)
    z = torch.complex(a, b)

    return z


def complex_dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    # work around unimplemented dropout for complex
    if input.is_complex():
        mask = torch.nn.functional.dropout(
            torch.ones_like(input.real),
            p,
            training,
        )
        return input * mask
    else:
        return torch.nn.functional.dropout(input, p, training)


def whiten22(
    X,
    training=True,
    running_mean=None,
    running_cov=None,
    momentum=0.1,
    eps=1e-5,
):

    tail = 1, X.shape[2], *([1] * (X.dim() - 3))
    axes = 1, *range(3, X.dim())

    if training or running_mean is None:
        mean = X.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)

    else:
        mean = running_mean

    X = X - mean.reshape(2, *tail)
    if training or running_cov is None:
        # stabilize by a small ridge
        var = (X * X).mean(dim=axes) + eps
        cov_uu, cov_vv = var[0], var[1]

        # has to mul-mean here anyway (naïve) : reduction axes shifted left.
        cov_vu = cov_uv = (X[0] * X[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack(
                [
                    cov_uu.data,
                    cov_uv.data,
                    cov_vu.data,
                    cov_vv.data,
                ],
                dim=0,
            ).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)

    else:
        cov_uu, cov_uv, cov_vu, cov_vv = running_cov.reshape(4, -1)

    sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    # torch.det uses svd, so may yield -ve machine zero

    denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

    # 4. apply Q to x (manually)
    z = torch.stack(
        [
            X[0] * p.reshape(tail) + X[1] * r.reshape(tail),
            X[0] * q.reshape(tail) + X[1] * s.reshape(tail),
        ],
        dim=0,
    )

    return z


def complex_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=True,
    momentum=0.1,
    eps=1e-05,
):
    assert (running_mean is None and running_var is None) or (
        running_mean is not None and running_var is not None
    )
    assert (weight is None and bias is None) or (
        weight is not None and bias is not None
    )

    X = torch.view_as_real_copy(input)
    X = torch.permute(X, (len(X.shape) - 1, *range(len(X.shape) - 1)))

    z = whiten22(
        X,
        training=training,
        running_mean=running_mean,
        running_cov=running_var,
        momentum=momentum,
        eps=eps,
    )

    if weight is not None and bias is not None:
        shape = 1, X.shape[2], *([1] * (X.dim() - 3))
        weight = weight.reshape(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.reshape(2, *shape)

    z = torch.permute(z, (*range(1, len(z.shape)), 0)).contiguous()
    z = torch.view_as_complex(z)

    return z
