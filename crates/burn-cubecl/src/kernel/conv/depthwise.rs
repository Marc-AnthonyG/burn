//! A depthwise 2D convolution: one input channel per group, one output channel
//! per group.
//!
//! The direct kernel vectorizes over the channels of one group, and a depthwise
//! group has one, so it runs every such convolution one channel per lane with
//! a scalar read per tap. Here a lane takes a vector of neighbouring channels
//! instead: in NHWC they sit together in memory, every one of them reads the
//! same taps, and the weights are laid out `[kh, kw, c]` so a tap's weights for
//! those channels are one vector too.

use crate::{
    kernel::{into_contiguous_aligned, utils::address_type, utils::decompose_linear},
    ops::{max_vector_size, numeric::empty_device_dtype, permute},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearViewMut,
};
use cubecl::{num_traits::Zero, std::FastDivmod};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeLaunch, CubeType)]
struct DepthwiseArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: i32,
    padding_w: i32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn depthwise_conv2d_kernel<E: Numeric, N: Size>(
    input: &Tensor<Vector<E, N>>,
    weight: &Tensor<Vector<E, N>>,
    bias: ComptimeOption<&[Vector<E, N>]>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    args: DepthwiseArgs,
    shape_out: Sequence<FastDivmod<u32>>,
    shape_out_c: FastDivmod<u32>,
    #[comptime] has_padding: bool,
    #[define(E)] _dtype: ElemType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let vector_size = output.vector_size();
    let pos = ABSOLUTE_POS * vector_size;

    let (rem, c) = shape_out_c.div_mod(pos as u32);
    let (b, spatial_pos) = decompose_linear(rem, &shape_out);
    let out_y = *spatial_pos.index(0);
    let out_x = *spatial_pos.index(1);

    let channels = input.shape(3);
    let in_batch_offs = b as usize * input.stride(0);

    let bias: ComptimeOption<Vector<E, N>> = bias.map(|bias| bias[c as usize / vector_size]);
    let mut sum = bias.unwrap_or_else(|| Vector::zero());

    for ky in 0..args.kernel_h {
        let in_y = (out_y * args.stride_h + ky * args.dilation_h) as i32 - args.padding_h;
        let mut row_in_bounds = true;
        if has_padding {
            row_in_bounds = in_y >= 0 && (in_y as u32) < args.in_h;
        }

        for kx in 0..args.kernel_w {
            let in_x = (out_x * args.stride_w + kx * args.dilation_w) as i32 - args.padding_w;
            let mut in_bounds = row_in_bounds;
            if has_padding {
                in_bounds &= in_x >= 0 && (in_x as u32) < args.in_w;
            }

            if in_bounds {
                let in_pos = in_batch_offs
                    + in_y as usize * input.stride(1)
                    + in_x as usize * input.stride(2)
                    + c as usize;
                let weight_pos = (ky * args.kernel_w + kx) as usize * channels + c as usize;

                sum += input[in_pos / vector_size] * weight[weight_pos / vector_size];
            }
        }
    }

    output.write(ABSOLUTE_POS, sum);
}

/// Whether a convolution is depthwise: every group one channel in and one out.
pub(crate) fn is_depthwise<const N: usize>(
    input: &CubeTensor,
    weight: &CubeTensor,
    options: &ConvOptions<N>,
) -> bool {
    let rank = input.meta.shape().num_dims();
    let in_channels = input.meta.shape()[rank - 1];
    let out_channels = weight.meta.shape()[0];
    let in_channels_per_group = weight.meta.shape()[rank - 1];

    N == 2
        && options.groups == in_channels
        && options.groups == out_channels
        && in_channels_per_group == 1
}

/// The depthwise convolution, on NHWC tensors whose channel dimension is
/// contiguous, as [`conv_direct`](super::conv_direct) hands them over.
pub(crate) fn conv_depthwise<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_shape = input.meta.shape()[1..dim_c].to_vec();
    let channels = weight.meta.shape()[0];
    let kernel_shape = weight.meta.shape()[1..dim_c].to_vec();

    // `[c, kh, kw, 1]` to `[kh, kw, c]`, so a tap's weights for neighbouring
    // channels are neighbours in memory.
    let weight = into_contiguous_aligned(permute(weight, &[1, 2, 0, 3]));

    let out_size = calculate_conv_output_sizes(
        &kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        &in_shape,
    );
    let has_padding =
        super::direct::should_check_spatial_bounds(&in_shape, &kernel_shape, &out_size, &options);

    let mut shape_out = vec![batch_size];
    shape_out.extend(out_size.iter().copied());
    shape_out.push(channels);

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.into(),
        out_dtype,
    );

    let vector_size = max_vector_size(&input).min(max_vector_size(&weight));

    let shape_out = output.meta.shape()[1..dim_c]
        .iter()
        .map(|s| *s as u32)
        .collect();
    let shape_out_c = channels as u32;

    let args = DepthwiseArgsLaunch::new(
        options.stride[0] as u32,
        options.stride[1] as u32,
        options.dilation[0] as u32,
        options.dilation[1] as u32,
        options.padding_begin()[0] as i32,
        options.padding_begin()[1] as i32,
        in_shape[0] as u32,
        in_shape[1] as u32,
        kernel_shape[0] as u32,
        kernel_shape[1] as u32,
    );

    let working_units = output.meta.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    unsafe {
        depthwise_conv2d_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(input, weight, bias, output),
            vector_size,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(),
            output.clone().into_linear_view(),
            args,
            shape_out,
            shape_out_c,
            has_padding,
            dtype_to_storage_type(out_dtype),
        )
    };

    Ok(output)
}
