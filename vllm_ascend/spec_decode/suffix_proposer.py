import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.spec_decode.suffix_decoding import \
    SuffixDecodingProposer as VllmSuffixDecodingProposer

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType

from arctic_inference.suffix_decoding import SuffixDecodingDraft, SuffixDecodingCache



class SuffixDecodingProposer(VllmSuffixDecodingProposer, Proposer):

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config)
        self.name = SpecDcodeType.SUFFIX
        self.device = device
        self.runner = runner

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens,
                  with_prefill=None,
                  in_graph_capturing=None,
                  num_reqs=None,
                  num_tokens_across_dp=None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None,
                  is_profile=False):
        pass

    def generate_token_ids(self,
                           valid_sampled_token_ids,
                           sampling_metadata=None,
                           scheduler_output=None,
                           spec_decode_metadata=None,
                           positions=None,
                           num_scheduled_tokens=None,
                           hidden_states=None,
                           aux_hidden_states=None) -> list[list[int]]:
        draft_token_ids = self.propose(self.runner.input_batch,
                                       valid_sampled_token_ids)
        return draft_token_ids
    
    def propose_suffix_draft(
        self,
        sampled_token_ids: list[list[int]],
        input_batch: InputBatch,
        _suffix_cache: SuffixDecodingCache,
    ) -> list[SuffixDecodingDraft]:
        results = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                results.append(SuffixDecodingDraft())
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                results.append(SuffixDecodingDraft())
                continue

            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            pattern = pattern.tolist()
            result = _suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(self.num_speculative_tokens,
                                    self.max_model_len - num_tokens - 1),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob)

            results.append(result)

        return results
