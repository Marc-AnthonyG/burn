use crate::engine::codegen::{
    io::{ref_buffer_len, ref_len, ref_vector_size},
    ir::{FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsExpand, LocalArgs, LocalArgsExpand},
    kernel::{fuse_on_read, fuse_on_write, init_locals},
};
use cubecl::prelude::*;
use cubek::reduce::components::args::{ReduceArgs, ReduceDType};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct FusedReduceArgs;

/// How the reduce routines see the reference layout.
///
/// The routines reduce one axis, so a reduction over several dims that sit
/// next to each other in memory is shown to them as a view in which that run
/// is one axis. Every view axis names the reference dims it stands for, outer
/// to inner, and the dim whose stride it carries. The identity view lists
/// every dim on its own, which is what a single-axis reduction uses.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ReduceView {
    pub axes: Vec<ViewAxis>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewAxis {
    pub dims: Vec<usize>,
    pub stride_dim: usize,
}

/// A reduction over several dims as the routines see it: the view in which
/// those dims are one axis, that axis, and the view's shape and strides.
pub struct MergedReduceView {
    pub view: ReduceView,
    pub axis: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl ReduceView {
    pub fn identity(rank: usize) -> Self {
        Self {
            axes: (0..rank)
                .map(|dim| ViewAxis {
                    dims: vec![dim],
                    stride_dim: dim,
                })
                .collect(),
        }
    }

    /// The view in which `reduced` is one axis, if those dims form one run of
    /// the layout given by `strides`.
    pub fn merged(
        shape: &[usize],
        strides: &[usize],
        reduced: &[usize],
    ) -> Option<MergedReduceView> {
        let rank = shape.len();
        let mut order: Vec<usize> = (0..rank).collect();
        order.sort_by(|&a, &b| strides[b].cmp(&strides[a]).then(a.cmp(&b)));

        let is_reduced = |dim: &usize| reduced.contains(dim);
        let matters: Vec<usize> = order
            .iter()
            .copied()
            .filter(|dim| shape[*dim] != 1)
            .collect();
        let run: Vec<usize> = matters.iter().copied().filter(is_reduced).collect();
        let positions: Vec<usize> = matters
            .iter()
            .enumerate()
            .filter(|(_, dim)| is_reduced(dim))
            .map(|(position, _)| position)
            .collect();
        if positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return None;
        }
        if run
            .windows(2)
            .any(|pair| strides[pair[0]] != strides[pair[1]] * shape[pair[1]])
        {
            return None;
        }

        let stride_dim = run.last().copied().unwrap_or(reduced[0]);
        let mut axes = Vec::with_capacity(rank);
        let mut run_axis = None;
        for dim in order {
            if is_reduced(&dim) {
                if run_axis.is_none() {
                    run_axis = Some(axes.len());
                    axes.push(ViewAxis {
                        dims: reduced.to_vec(),
                        stride_dim,
                    });
                }
            } else {
                axes.push(ViewAxis {
                    dims: vec![dim],
                    stride_dim: dim,
                });
            }
        }

        let view_shape: Vec<usize> = axes
            .iter()
            .map(|axis| axis.dims.iter().map(|dim| shape[*dim]).product())
            .collect();
        let view_strides: Vec<usize> = axes.iter().map(|axis| strides[axis.stride_dim]).collect();
        Some(MergedReduceView {
            view: Self { axes },
            axis: run_axis?,
            shape: view_shape,
            strides: view_strides,
        })
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct FusedReduceInput {
    pub global: GlobalArgs,
    #[cube(comptime)]
    pub config: FuseBlockConfig,
    #[cube(comptime)]
    pub arg: FuseArg,
    #[cube(comptime)]
    pub view: ReduceView,
}

#[derive(CubeType, CubeLaunch)]
pub struct FusedReduceOutput {
    pub global: GlobalArgs,
    #[cube(comptime)]
    pub config: FuseBlockConfig,
    #[cube(comptime)]
    pub arg: FuseArg,
    #[cube(comptime)]
    pub view: ReduceView,
}

#[derive(Clone)]
pub struct FusedReduceState {
    inputs: GlobalArgs,
    outputs: GlobalArgs,
    locals_on_read: LocalArgs,
    locals_on_write: LocalArgs,
    config_on_read: FuseBlockConfig,
    config_on_write: FuseBlockConfig,
    // TODO: Should be a list when multiple blocks are there.
    input: FuseArg,
    out: FuseArg,
    view: ReduceView,
}

#[derive(Clone)]
pub struct FusedReduceStateExpand {
    inputs: GlobalArgsExpand,
    outputs: GlobalArgsExpand,
    locals_on_read: LocalArgsExpand,
    locals_on_write: LocalArgsExpand,
    config_on_read: FuseBlockConfig,
    config_on_write: FuseBlockConfig,
    input: FuseArg,
    out: FuseArg,
    view: ReduceView,
}

#[cube]
impl ReduceArgs for FusedReduceArgs {
    type Input<E: Numeric, S: Size> = FusedReduceInput;
    type Output<E: Numeric, S: Size> = FusedReduceOutput;
    type State<P: ReduceDType> = FusedReduceState;

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In, P::SizeIn>,
        output: &mut Self::Output<P::Out, P::SizeOut>,
    ) -> Self::State<P> {
        let mut locals_read = init_locals(&input.global, &mut output.global, &input.config);
        let mut locals_write = init_locals(&input.global, &mut output.global, &output.config);
        // TODO Add stuff from previous blocks to the local of each block.
        FusedReduceState::new(input, output, &mut locals_read, &mut locals_write)
    }

    fn read_input<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::In, P::SizeIn> {
        let mut state = state.clone();
        let value = fuse_on_read::<P::In, P::SizeIn>(
            &state.inputs,
            &mut state.outputs,
            &mut state.locals_on_read,
            index,
            comptime! {
                let mut sequence = Sequence::new();
                // TODO: Register local arguments from previous blocks.
                sequence.push(state.input.clone());
                sequence
            },
            &state.config_on_read,
        );
        value[0]
    }

    fn read_output<P: ReduceDType>(
        _state: &Self::State<P>,
        _index: usize,
    ) -> Vector<P::Out, P::SizeOut> {
        Vector::empty()
    }

    fn write_output<P: ReduceDType>(
        state: &mut Self::State<P>,
        index: usize,
        value: Vector<P::Out, P::SizeOut>,
    ) {
        let mut values = Registry::<FuseArg, Vector<P::Out, P::SizeOut>>::new();
        let mut args = comptime![Vec::<FuseArg>::new()];

        values.insert(comptime![state.out.clone()], value);
        comptime![args.push(state.out.clone())];
        fuse_on_write(
            &state.inputs,
            &mut state.outputs,
            &mut state.locals_on_write,
            index,
            values,
            args,
            &state.config_on_write,
        );
    }

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        ref_len(
            &state.inputs,
            &state.outputs,
            &state.locals_on_read,
            &state.config_on_read,
        )
    }

    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        ref_len(
            &state.inputs,
            &state.outputs,
            &state.locals_on_write,
            &state.config_on_write,
        )
    }

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        ref_buffer_len(
            &state.inputs,
            &state.outputs,
            &state.locals_on_read,
            &state.config_on_read,
        )
    }

    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        ref_buffer_len(
            &state.inputs,
            &state.outputs,
            &state.locals_on_write,
            &state.config_on_write,
        )
    }

    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        let rank = comptime![state.view.axes.len()];
        rank.runtime()
    }

    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        let rank = comptime![state.view.axes.len()];
        rank.runtime()
    }

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        view_shape(&state.locals_on_read, comptime![state.view.clone()], dim)
    }

    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        view_shape(&state.locals_on_write, comptime![state.view.clone()], dim)
    }

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        view_stride(&state.locals_on_read, comptime![state.view.clone()], dim)
    }

    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        view_stride(&state.locals_on_write, comptime![state.view.clone()], dim)
    }

    fn vector_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        ref_vector_size(&state.locals_on_read)
    }

    fn vector_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        ref_vector_size(&state.locals_on_write)
    }
}

#[cube]
impl FusedReduceState {
    pub fn new(
        inputs: &FusedReduceInput,
        outputs: &mut FusedReduceOutput,
        locals_on_read: &mut LocalArgs,
        locals_on_write: &mut LocalArgs,
    ) -> FusedReduceState {
        FusedReduceState {
            inputs: inputs.global.clone(),
            outputs: outputs.global.clone(),
            locals_on_read: locals_on_read.clone(),
            locals_on_write: locals_on_write.clone(),
            config_on_read: comptime![inputs.config.clone()],
            config_on_write: comptime![outputs.config.clone()],
            input: comptime![inputs.arg.clone()],
            out: comptime![outputs.arg.clone()],
            view: comptime![inputs.view.clone()],
        }
    }
}

/// The reference shape of a view axis: the product over the dims it stands for.
#[cube]
#[allow(clippy::needless_range_loop)]
fn view_shape(locals: &LocalArgs, #[comptime] view: ReduceView, dim: usize) -> usize {
    let rank = comptime![view.axes.len()];
    let mut result = 0;

    #[unroll]
    for axis in 0..rank {
        if dim == axis {
            let dims = comptime![view.axes[axis].dims.clone()];
            let count = comptime![dims.len()];
            let mut product = 1;

            #[unroll]
            for i in 0..count {
                let reference_dim = comptime![dims[i]];
                product *= locals.ref_shape[reference_dim];
            }

            result = product;
        }
    }

    result
}

/// The reference stride of a view axis: that of its innermost dim.
#[cube]
fn view_stride(locals: &LocalArgs, #[comptime] view: ReduceView, dim: usize) -> usize {
    let rank = comptime![view.axes.len()];
    let mut result = 0;

    #[unroll]
    for axis in 0..rank {
        if dim == axis {
            let reference_dim = comptime![view.axes[axis].stride_dim];
            result = locals.ref_strides[reference_dim];
        }
    }

    result
}

impl CubeType for FusedReduceState {
    type ExpandType = FusedReduceStateExpand;
}

impl IntoExpand for FusedReduceStateExpand {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl ExpandTypeClone for FusedReduceStateExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl AsRefExpand for FusedReduceStateExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl AsMutExpand for FusedReduceStateExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl IntoMut for FusedReduceStateExpand {
    fn into_mut(self, _context: &Scope) -> Self {
        self
    }
}

impl CubeDebug for FusedReduceStateExpand {}
