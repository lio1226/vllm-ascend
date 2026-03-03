import collections
import math
from contextlib import contextmanager
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch_npu

from vllm.v1.outputs import SamplerOutput
from vllm.model_executor.models.arctic_speculator import (
    ArcticLSTMSpeculator,
    SQRT2
)


def _generate_cg_key(padding_size: int, head_index: int):
    return (padding_size << 16) + head_index


@contextmanager
def graph_capture(device):
    """NPU ACL Graph capture context manager."""
    stream = torch.npu.Stream(device=device)
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    class _GraphCaptureContext:
        pass

    context = _GraphCaptureContext()
    context.stream = stream
    with torch.npu.stream(stream):
        yield context

class AscendArcticLSTMSpeculator(ArcticLSTMSpeculator):

    def generate_states(
        self,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        head_index: int,
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:

        if head_index == 0 and self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2

        actual_i = 0 if self.tie_weights else head_index
        actual_proj_i = 1 if self.tie_weights and head_index >= 2 else head_index

        if self.method == "sum_lstm":
            assert self.tie_lstm_embs

            prev_state = previous_hidden_states

            z4 = self.forget_emb[actual_i](last_tokens).repeat(1, 1, 4)  # b n d
            states = self.projs[actual_proj_i](prev_state)

            if False:
                # Shapes:
                #   prev_state: [B, 1, D_eff] (e.g., 2880 in the first round and 4096 later)
                #   states:     [B, 1, 4*D_gate] (e.g., 4*4096)
                #   z4:         [B, 1, 4*D_gate]
                states_4d = states.flatten(0, 1).contiguous()  # [B, 4*D_gate]
                z4_4d     = z4.flatten(0, 1).contiguous()      # [B, 4*D_gate]

                orig_cell_shape = cell_states.shape           # [B, 1, D_gate]
                pc_d  = cell_states.flatten(0, 1).contiguous()  # [B, D_gate]
                
                state_d = torch.empty_like(pc_d)
                cell_d = torch.empty_like(pc_d)
                # Optional precondition checks that mirror the kernel's TORCH_CHECKs:
                assert states_4d.size(-1) % 4 == 0
                assert z4_4d.size(-1) == states_4d.size(-1)
                assert pc_d.size(-1) == states_4d.size(-1) // 4

                w_cell  = self.cell_ln[actual_i].weight
                b_cell  = self.cell_ln[actual_i].bias
                w_state = self.state_ln[actual_i].weight
                b_state = self.state_ln[actual_i].bias

                alpha      = float(self.emb_weight / self.state_weight)
                eps_cell   = float(self.cell_ln[actual_i].eps)
                eps_state  = float(self.state_ln[actual_i].eps)
                use_fast_gelu = False

                sum_lstm_graph(
                    states_4d, z4_4d, pc_d,
                    w_cell, b_cell, w_state, b_state,
                    state_d, cell_d,
                    alpha, eps_cell, eps_state
                )
                # state_d, cell_d = torch.ops._C_ascend.npu_sum_lstm(
                #         states_4d, z4_4d, pc_d,
                #         w_cell, b_cell, w_state, b_state,
                #         alpha, eps_cell, eps_state, use_fast_gelu)

                state       = state_d.reshape(orig_cell_shape)    # [B, 1, D_gate]
                cell_states = cell_d.reshape(orig_cell_shape)     # [B, 1, D_gate]

                return state, cell_states
            else:
                added_states = torch.add(states,
                                        z4,
                                        alpha=self.emb_weight / self.state_weight)

                forget_input_output, cell_candidate = added_states.split(
                    [self.proj_dim[0] * 3, self.proj_dim[0]], dim=-1)
                forget_gate, input_gate, output_gate = torch.sigmoid(
                    forget_input_output).split(
                        [self.proj_dim[0], self.proj_dim[0], self.proj_dim[0]],
                        dim=-1)

                cell_candidate = self.activation(
                    self.cell_ln[actual_i](cell_candidate))  # b n d
                cell_candidate = cell_candidate * input_gate

                cell_states = cell_states * forget_gate
                cell_states = cell_states + cell_candidate

                state_candidate = self.activation(
                    self.state_ln[actual_i](cell_states))
                state = state_candidate * output_gate

                return state, cell_states

        else:
            # Project and predict
            z4 = self.emb[actual_i](last_tokens)  # b k d
            states = self.proj[actual_proj_i](previous_hidden_states)

            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            states.add_(z4, alpha=self.emb_weight / self.state_weight)
            states = self.activation(self.ln[actual_i](states))  # b k d

            return states


    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> List[SamplerOutput]:
        
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(f"Max speculative tokens for model is "
                             f"{self.max_speculative_tokens}, but "
                             f"{num_predict_tokens} were requested")

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        batch_size = input_ids.size(0)
        state_shapes = list(previous_hidden_states.shape)
        state_shapes[-1] = self.inner_dim[-1]

        static_next_tokens = [None] * num_predict_tokens
        static_cell_states = None
        static_last_tokens = None
        static_hidden_states = None

        static_states = self.static_cuda_buffers["previous_hidden_states"]
        if self.method == "sum_lstm":
            previous_cell_states = torch.zeros(
                state_shapes,
                device=previous_hidden_states.device,
                dtype=previous_hidden_states.dtype,
            )
            (
                padded_size,
                static_last_tokens,
                static_hidden_states,
                static_cell_states,
            ) = self._prepare_cuda_graph_ios(
                batch_size,
                last_tokens,
                previous_hidden_states,
                static_states,
                previous_cell_states,
                use_lstm=True,
            )
        else:
            padded_size, static_last_tokens, static_hidden_states = (
                self._prepare_cuda_graph_ios(batch_size, last_tokens,
                                             previous_hidden_states,
                                             static_states))
        if self.cuda_graph_mode and batch_size <= self.cuda_graph_max_batch_size:
            cg_key = _generate_cg_key(padded_size, 0)
            g = self.cuda_graphs.get(cg_key)

            static_states = (
                self.static_cuda_buffers["next_previous_hidden_states"]
                if self.inner_dim[-1] != self.input_hidden_dim else
                self.static_cuda_buffers["previous_hidden_states"])

            for i in range(num_predict_tokens):
                static_next_tokens[i] = self.static_cuda_buffers[
                    "next_tokens"][i][:padded_size]

            if g is None:
                device = torch.npu.current_device()
                for i in range(num_predict_tokens):
                    self.static_cuda_buffers["next_tokens"][i][:padded_size] = torch.zeros(
                        (padded_size, 1), dtype=torch.long, device=device)
                with graph_capture(device=device) as capture_context:
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g):
                        if self.method == "sum_lstm":
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                                cell_states=static_cell_states,
                            )
                        else:
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                            )
                self.cuda_graphs[cg_key] = g
            else:
                g.replay()
        else:
            if self.method == "sum_lstm":
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                    cell_states=static_cell_states,
                )
            else:
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                )

        next_tokens = []
        for i in range(num_predict_tokens):
            next_tokens.append(static_next_tokens[i][:batch_size])

        return torch.cat(next_tokens, dim=-1)

    def generate_token_ids(
        self,
        batch_size: int,
        num_predict_tokens: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_tokens_tensors: List[torch.Tensor],
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:
        for head_index in range(num_predict_tokens):
            if self.method == "sum_lstm":
                states, cell_states = self.generate_states(
                    last_tokens, previous_hidden_states, head_index,
                    cell_states)
            else:
                states = self.generate_states(last_tokens,
                                              previous_hidden_states,
                                              head_index)
            previous_hidden_states = states
            states = states.flatten(0, 1)
            head_weight = (self.qhead[head_index] if self.qhead is not None
                           and batch_size <= 32 else self.head[head_index])
            logits = self.logits_processor(head_weight, states)

            if self.tp_size == 1:
                last_tokens = torch.argmax(logits,
                                           dim=-1).reshape(batch_size, -1)
            else:
                vals, indices = torch.topk(logits, 1, dim=-1)
                indices = indices + self.tp_rank * logits.shape[-1]

                packed_data = torch.cat(
                    [vals.to(torch.float64).view(torch.int32), indices], dim=0)
                packed_data = self.TP_GROUP.all_gather(packed_data)
                vals, indices = packed_data.split(batch_size, dim=0)
                vals = vals.view(torch.float64)

                argidx = torch.argmax(vals, -1).reshape(batch_size, -1)
                last_tokens = torch.gather(indices, -1, argidx)

            if next_tokens_tensors[head_index] == None:
                next_tokens_tensors[head_index] = last_tokens
            else:
                next_tokens_tensors[head_index].copy_(last_tokens)

        return next_tokens_tensors

ArcticLSTMSpeculator.generate_states = AscendArcticLSTMSpeculator.generate_states
ArcticLSTMSpeculator.generate_proposals = AscendArcticLSTMSpeculator.generate_proposals
ArcticLSTMSpeculator.generate_token_ids = AscendArcticLSTMSpeculator.generate_token_ids