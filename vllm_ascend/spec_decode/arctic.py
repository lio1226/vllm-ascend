# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

from vllm.config import VllmConfig, CUDAGraphMode
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.logger import init_logger

import numpy as np
import torch

from vllm.v1.spec_decode.arctic import padding_size
from vllm.v1.spec_decode.arctic import ArcticProposer as VllmArcticProposer
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType


from vllm_ascend.ascend_forward_context import set_ascend_forward_context
logger = init_logger(__name__)

class ArcticProposer(VllmArcticProposer, Proposer):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)

        self.name = SpecDcodeType.ARCTIC
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = int(
            vllm_config.speculative_config.draft_model_config.hf_text_config.input_hidden_dim
        )
        self.dtype = vllm_config.model_config.dtype

    def propose_draft_token_ids(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> Optional[np.ndarray]:
        draft_token_ids = self.propose(
            context_token_ids=context_token_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens
            )
        
        return draft_token_ids
    
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False
    ) -> None:
        num_predict_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        size = padding_size(self.vllm_config.scheduler_config.max_num_seqs)
        input_ids = torch.rand(size, dtype=self.dtype)
        previous_hidden_states = torch.zeros(
            (size, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_ascend_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_actual_tokens=0,
                in_profile_run=is_profile,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                is_draft_model=True):
            self.model.generate_proposals(
                input_ids=input_ids,
                previous_hidden_states=previous_hidden_states,
                num_predict_tokens=num_predict_tokens,
            )