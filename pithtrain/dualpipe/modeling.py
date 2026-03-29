from dataclasses import fields
from typing import List, Optional

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed

from pithtrain.dualpipe.execution import (
    IntermediateTensorsLayer,
    Stage1Args,
    Stage1OutsMlp,
    Stage1OutsMoe,
    Stage1Record,
    Stage2Args,
    Stage2Outs,
    Stage2Record,
    Stage3Args,
    Stage3Outs,
    Stage3Record,
    Stage4Args,
    Stage4Outs,
    Stage4Record,
    Stage5Args,
    Stage5Outs,
    Stage5Record,
)
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import ModelImplMode
from pithtrain.models.interface import DecoderLayerProtocol
from pithtrain.operators.all_to_all import direct_all_to_all


def decoder_layer_forward_dispatch(
    sorted_tokens: torch.Tensor,
    output_splits: Optional[List[int]],
    input_splits: Optional[List[int]],
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    All-to-all dispatch.
    """
    if output_splits is not None:
        gathered_tokens = direct_all_to_all(
            sorted_tokens,
            output_splits,
            input_splits,
            ep_group,
        )
        a2a_ctx = (output_splits, input_splits, ep_group)
    else:
        gathered_tokens = sorted_tokens
        a2a_ctx = None
    return gathered_tokens, a2a_ctx


def decoder_layer_forward_combine(
    outs: torch.Tensor,
    input_splits: Optional[List[int]],
    output_splits: Optional[List[int]],
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    All-to-all combine.
    """
    if output_splits is not None:
        outs = direct_all_to_all(
            outs,
            input_splits,
            output_splits,
            ep_group,
        )
        a2a_ctx = (input_splits, output_splits, ep_group)
    else:
        a2a_ctx = None
    return outs, a2a_ctx


def decoder_layer_forward(
    layer: DecoderLayerProtocol,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
):
    """
    Forward pass for a DualPipeV decoder layer.
    """

    if ModelImplMode.use_reference_fwd:
        return (
            layer.reference_forward(hidden_states, position_ids),
            [],
        )

    intermediate_tensors = IntermediateTensorsLayer()

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()
    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    output = layer.forward_attn(next_hidden_states, position_ids)
    (
        hidden_states,
        sorted_tokens,
        moe_local_idxs,
        topk_weight,
        output_splits,
        input_splits,
        expert_idxs,
        residual,
        expand_idx,
        dedup_input_splits,
        dedup_output_splits,
    ) = output

    has_experts = hasattr(layer.mlp, "experts")
    ep_group = layer.mlp.ep_group if has_experts else None

    if has_experts:
        record.outs = Stage1OutsMoe(
            output.hidden_states, output.sorted_tokens, output.topk_weight, output.residual
        )
    else:
        record.outs = Stage1OutsMlp(output.sorted_tokens, output.residual)
    intermediate_tensors.stage1 = record
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()
    sorted_tokens = sorted_tokens.detach().requires_grad_()
    record.args = Stage2Args(sorted_tokens)

    gathered_tokens, record.ctx = decoder_layer_forward_dispatch(
        sorted_tokens, dedup_output_splits, dedup_input_splits, ep_group
    )
    fwd_comm_work = getattr(gathered_tokens, "comm_work", None)
    setattr(gathered_tokens, "comm_work", None)

    record.outs = Stage2Outs(gathered_tokens)
    intermediate_tensors.stage2 = record
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()
    gathered_tokens = gathered_tokens.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens)

    if fwd_comm_work is not None:
        fwd_comm_work.wait()

    moe_outs = layer.forward_mlp(gathered_tokens, expert_idxs, expand_idx)

    record.outs = Stage3Outs(moe_outs)
    intermediate_tensors.stage3 = record
    nvtx.range_pop()

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()
    moe_outs = moe_outs.detach().requires_grad_()
    record.args = Stage4Args(moe_outs)

    moe_outs, record.ctx = decoder_layer_forward_combine(
        moe_outs, input_splits, output_splits, ep_group
    )
    fwd_comm_work = getattr(moe_outs, "comm_work", None)
    setattr(moe_outs, "comm_work", None)

    record.outs = Stage4Outs(moe_outs)
    intermediate_tensors.stage4 = record
    nvtx.range_pop()

    # Stage 5.
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()
    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    hidden_states = hidden_states.detach().requires_grad_()
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, hidden_states, residual)

    if fwd_comm_work is not None:
        fwd_comm_work.wait()
    hidden_states = layer.forward_aggregate(
        moe_outs, moe_local_idxs, topk_weight, hidden_states, residual
    )

    record.outs = Stage5Outs(hidden_states)
    intermediate_tensors.stage5 = record
    nvtx.range_pop()

    return hidden_states, intermediate_tensors


def decoder_layer_backward(
    layer: DecoderLayerProtocol,
    dy: Optional[List[torch.Tensor]],
    loss: Optional[torch.Tensor],
    intermediate_tensors_layer: IntermediateTensorsLayer,
):
    """
    Backward pass for a DualPipeV decoder layer.

    Handles both normal and merged cases using asymmetric None pattern:
    - Merged stage1: stage1.outs is set, stage1.args is None
      -> Run backward on stage1.outs, grads flow to prev layer's stage5.args
      -> Return None to signal prev layer to get grads from stage5.args
    - Merged stage5: stage5.args is set, stage5.outs is None
      -> Get grads from stage5.args.*.grad (already computed by next layer)
    """

    # Check if this layer's stage5 was merged with the NEXT layer's stage1.
    # Detection: stage5.args is set, stage5.outs is None
    stage5_record = intermediate_tensors_layer.stage5
    stage5_was_merged = (
        hasattr(stage5_record, "args")
        and stage5_record.args is not None
        and not (hasattr(stage5_record, "outs") and stage5_record.outs is not None)
    )

    # Check if this layer's stage1 is merged with the PREVIOUS layer's stage5.
    # Detection: stage1.outs is set, stage1.args is None
    stage1_record = intermediate_tensors_layer.stage1
    stage1_is_merged = (
        hasattr(stage1_record, "outs")
        and stage1_record.outs is not None
        and not (hasattr(stage1_record, "args") and stage1_record.args is not None)
    )

    # Stage 5.
    if loss is not None:
        assert False, "loss should not be provided"
        loss.backward()
        loss.detach_()
    elif stage5_was_merged:
        # Stage5 was merged with next layer's stage1. Get grads from stage5.args.
        # These were computed when the next layer ran its merged stage1 backward.
        nvtx.range_push("layer%02d.stage5_merged_skip" % layer.idx)
        moe_outs_grad, topk_weight_grad, hidden_states_grad, residual_grad = [
            t.grad if t is not None else None for t in stage5_record.args
        ]
        nvtx.range_pop()
    else:
        # Normal case: run stage5 backward
        nvtx.range_push("layer%02d.stage5_b" % layer.idx)
        record = stage5_record
        run_backward(record.outs, dy)
        moe_outs_grad, topk_weight_grad, hidden_states_grad, residual_grad = [
            t.grad if t is not None else None for t in record.args
        ]
        nvtx.range_pop()

    # Some models do not have shared experts, so the original hidden states weren't used
    # during the forward pass of stage 5. As a result, its gradients will be None. For
    # compatibility with the pipeline, we manually set its gradient to zero here.
    if hidden_states_grad is None:
        outs = stage1_record.outs
        data = outs.hidden_states if isinstance(outs, Stage1OutsMoe) else outs.sorted_tokens
        hidden_states_grad = torch.zeros_like(data)

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)
    record = intermediate_tensors_layer.stage4
    if record.ctx is not None:
        output_splits, input_splits, group = record.ctx
        moe_outs_grad = direct_all_to_all(moe_outs_grad, input_splits, output_splits, group)
        bwd_comm_work = moe_outs_grad.comm_work
        moe_outs_grad.comm_work = None
    else:
        bwd_comm_work = None
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)
    record = intermediate_tensors_layer.stage3

    if bwd_comm_work is not None:
        bwd_comm_work.wait()

    run_backward(record.outs, (moe_outs_grad,))
    gathered_tokens_grad = record.args.gathered_tokens.grad
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)
    record = intermediate_tensors_layer.stage2
    if record.ctx is not None:
        output_splits, input_splits, group = record.ctx
        sorted_tokens_grad = direct_all_to_all(
            gathered_tokens_grad, input_splits, output_splits, group
        )
        bwd_comm_work = sorted_tokens_grad.comm_work
        sorted_tokens_grad.comm_work = None
    else:
        sorted_tokens_grad = gathered_tokens_grad
        bwd_comm_work = None
    nvtx.range_pop()

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)
    if bwd_comm_work is not None:
        bwd_comm_work.wait()

    if hasattr(layer.mlp, "experts"):
        grad_tensors = (hidden_states_grad, sorted_tokens_grad, topk_weight_grad, residual_grad)
    else:
        grad_tensors = (sorted_tokens_grad, residual_grad)

    if stage1_is_merged:
        # Merged case: this layer's stage1 + previous layer's stage5
        # Run backward through stage1.outs. Grads flow to prev layer's stage5.args.
        run_backward(stage1_record.outs, grad_tensors)
        nvtx.range_pop()

        # Clear tensor refs but keep pre-allocated records
        for field in fields(intermediate_tensors_layer):
            record = getattr(intermediate_tensors_layer, field.name)
            for rf in fields(record):
                setattr(record, rf.name, None)

        # Return None to signal prev layer to get grads from its stage5.args
        return None
    else:
        # Normal case: run stage1 backward
        record = stage1_record
        run_backward(record.outs, grad_tensors)
        hidden_states_grad = record.args.next_hidden_states.grad
        record.args.prev_hidden_states.grad = hidden_states_grad
        nvtx.range_pop()

        # Clear tensor refs but keep pre-allocated records
        for field in fields(intermediate_tensors_layer):
            record = getattr(intermediate_tensors_layer, field.name)
            for rf in fields(record):
                setattr(record, rf.name, None)

        return hidden_states_grad
