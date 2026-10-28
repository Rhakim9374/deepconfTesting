"""
DeepThinkLLM implementation with online and offline mode support

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import itertools
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import copy

from .eet_asc import shadow_record
from .eet_voting import eet_voting_results
from .outputs import DeepThinkOutput
from .utils import (
    DEEPCONF_TOPK, READOUT_TOPK,
    process_batch_results, process_batch_results_offline, 
    weighted_majority_vote, compute_all_voting_results
)
from .processors import WrappedPerReqLogitsProcessor


@dataclass
class ObservedCompletion:
    """One request's output accumulated over the engine's delta outputs
    (the ``CompletionOutput`` fields ``process_output`` reads, vLLM's
    logprob dicts included), plus the step in which it finished."""
    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    logprobs: List[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    finish_time_s: Optional[float] = None
    batch_tokens_at_finish: Optional[int] = None


@dataclass
class ObservedRequest:
    outputs: List[ObservedCompletion]


def _multi_gpu_vllm_defaults(tensor_parallel_size: int) -> dict:
    """vLLM init kwargs that keep PCIe multi-GPU stable; empty for a
    single GPU. Mirror of ExploreExploitThink's
    ``backend._multi_gpu_vllm_defaults``.

    On a tensor-parallel split without NVLink, vLLM's custom all-reduce
    kernel hangs, so fall back to NCCL all-reduce; and the 0.9 memory
    default OOMs on the sampler-warmup spike, so leave headroom. NCCL's
    peer-to-peer and InfiniBand transports are disabled via the job
    environment (NCCL_P2P_DISABLE / NCCL_IB_DISABLE).
    """
    if tensor_parallel_size <= 1:
        return {}
    return {
        "disable_custom_all_reduce": True,
        "gpu_memory_utilization": 0.85,
    }


# Spaces, a blank line, non-ASCII and LaTeX, as in prompts and answers.
TOKENIZER_PROBE = ("Thus $\\boxed{1 - \\dfrac{2}{\\pi}}$.\n\nWait,  let me re-check: "
                   "x² ≥ 0 — 该助手 π\t\\left(-\\frac{13}{96},\\ \\frac{13}{40}\\right)")


def check_tokenizer(tokenizer, model_path: str) -> None:
    """Raise unless ``tokenizer`` encodes and decodes like the model's own
    ``tokenizer.json``. Mirror of ExploreExploitThink's
    ``backend.check_tokenizer``: Transformers 5 builds the tokenizer from
    the class ``tokenizer_config.json`` declares, and
    DeepSeek-R1-0528-Qwen3-8B declares LlamaTokenizerFast over a
    byte-level vocabulary, which strips spaces, newlines and non-ASCII
    from prompts and leaves byte markers (Ġ, Ċ) in the decoded text (fix:
    ExploreExploitThink's scripts/fix_tokenizer_class.sh)."""
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    ids = reference.encode(TOKENIZER_PROBE, add_special_tokens=False).ids
    if (tokenizer.encode(TOKENIZER_PROBE, add_special_tokens=False) != ids
            or tokenizer.decode(ids) != TOKENIZER_PROBE):
        raise RuntimeError(
            f"the tokenizer loaded for {model_path} does not reproduce its "
            "tokenizer.json")


class DeepThinkLLM:
    """Enhanced LLM wrapper with deep thinking capabilities"""
    
    def __init__(self, model: str, **vllm_kwargs):
        """
        Initialize DeepThinkLLM
        
        Args:
            model: Model path or name
            **vllm_kwargs: Additional arguments for vLLM initialization
        """
        self.model_name = model
        self.vllm_kwargs = vllm_kwargs
        
        # Initialize vLLM
        tp = vllm_kwargs.get(
            "tensor_parallel_size",
            len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        )
        default_kwargs = {
            "tensor_parallel_size": tp,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            # vLLM caps per-request logprobs at max_logprobs (default 20) and
            # raises VLLMValidationError above it, so the read-out request
            # has to be unlocked here; main.py passes utils.fetch_width.
            # ExploreExploitThink does the same in src/experiment.py.
            "max_logprobs": READOUT_TOPK,
            # Record raw log-softmax regardless of the request's
            # temperature/top-p/top-k; sampling still uses the processed
            # distribution. This is already vLLM's default (>= 0.10) and the
            # V1 engine's behaviour, so it is a defensive pin -- but it is
            # the pin that keeps readout_tail_conf the same quantity as the
            # other runner's tail_conf, which sets it too.
            "logprobs_mode": "raw_logprobs",
            # Gated-DeltaNet prefill (Qwen3.5/3.8) in Triton, as in
            # ExploreExploitThink's backend: on Hopper vLLM otherwise picks
            # FlashInfer's JIT-compiled kernel, and the nodes have no nvcc.
            "gdn_prefill_backend": "triton",
            # At most a few dozen sequences per batch; vLLM's H100 default
            # (1024) exceeds the per-sequence GDN state slots Qwen3.8 can
            # fit, which aborts CUDA-graph capture.
            "max_num_seqs": 256,
            # Engine statistics on, as in ExploreExploitThink: preemptions()
            # reads vLLM's counter and the job log gets the periodic
            # running/waiting/KV-usage line.
            "disable_log_stats": False,
            **_multi_gpu_vllm_defaults(tp),
        }
        default_kwargs.update(vllm_kwargs)
        
        print("Initializing vLLM engine...")
        llm_init_start = time.time()
        self.llm = LLM(model=model, logits_processors=[WrappedPerReqLogitsProcessor], **default_kwargs)
        check_tokenizer(self.llm.get_tokenizer(), model)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        # Store initialization times
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
        # Engine request ids must be unique while a request is live.
        self._request_ids = itertools.count()
    
    def preemptions(self) -> int:
        """vLLM's ``vllm:num_preemptions`` counter: zero growth over a
        question means every request stayed resident from admission to
        finish."""
        return sum(m.value for m in self.llm.get_metrics() if m.name == "vllm:num_preemptions")

    def generate(self, *args, **kwargs):
        """Simple wrapper around vLLM's generate method"""
        return self.llm.generate(*args, **kwargs)

    def generate_observed(self, prompts, params_list):
        """vLLM's ``generate`` path for one prompt-per-params batch,
        observed step by step (mirror of ExploreExploitThink's
        ``backend.observe_engine``): the requests are queued on the engine
        in delta-output mode, the engine loop runs here, and each result
        carries the elapsed time of the step in which it finished and the
        batch's decoded-token total at that step (requests finishing in
        the same step share both). Prompts go in as token ids, so both
        runners send identical tokens. Scheduling, sampling, the per-request
        logits processor and the returned content are unchanged; the
        results come back in prompt order with the ``.outputs`` shape
        ``process_batch_results`` reads.
        """
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind
        engine = self.llm.llm_engine
        tokenizer = self.llm.get_tokenizer()
        request_ids = []
        for prompt, params in zip(prompts, params_list):
            params = copy.deepcopy(params)
            params.output_kind = RequestOutputKind.DELTA
            rid = str(next(self._request_ids))
            request_ids.append(rid)
            # Token ids encoded as ExploreExploitThink's backend encodes them:
            # the chat template already writes the BOS, so none is added.
            engine.add_request(rid, TokensPrompt(prompt_token_ids=tokenizer.encode(prompt, add_special_tokens=False)), params)
        streams = {rid: ObservedCompletion() for rid in request_ids}
        decoded = 0
        start = time.perf_counter()
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            elapsed = time.perf_counter() - start
            finished = []
            for out in step_outputs:
                for completion in out.outputs:
                    target = streams[out.request_id]
                    target.text += completion.text
                    target.token_ids.extend(completion.token_ids)
                    if completion.logprobs:
                        target.logprobs.extend(completion.logprobs)
                    decoded += len(completion.token_ids)
                    if completion.finish_reason is not None:
                        target.finish_reason = completion.finish_reason
                        finished.append(target)
            for target in finished:
                target.finish_time_s = elapsed
                target.batch_tokens_at_finish = decoded
        return [ObservedRequest([streams[rid]]) for rid in request_ids]
    
    def deepthink(
        self,
        prompt: str,
        mode: str = "offline",
        # Online mode parameters
        warmup_traces: int = 16,
        total_budget: int = 256,
        confidence_percentile: int = 90,
        # Offline mode parameters  
        budget: int = 512,
        # Common parameters
        window_size: int = 2048,
        sampling_params: Optional[SamplingParams] = None,
        # Multiple voting options
        compute_multiple_voting: bool = True,
        **kwargs
    ) -> DeepThinkOutput:
        """
        Perform deep thinking on a prompt
        
        Args:
            prompt: Input prompt (prepared string)
            mode: "online" for confidence-based early stopping, "offline" for batch generation
            warmup_traces: Number of warmup traces for online mode
            total_budget: Total budget for online mode
            confidence_percentile: Percentile for confidence threshold in online mode
            budget: Number of traces for offline mode
            window_size: Window size for confidence computation
            sampling_params: Custom vLLM sampling parameters
            compute_multiple_voting: Whether to compute multiple voting method results
            
        Returns:
            DeepThinkOutput containing results
        """
        total_start_time = time.time()
        
        # Create output object
        output = DeepThinkOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=32000,
                logprobs=READOUT_TOPK,
            )
            
        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": mode,
            "window_size": window_size,
            "compute_multiple_voting": compute_multiple_voting,
        }
        
        if mode == "online":
            output.config.update({
                "warmup_traces": warmup_traces,
                "total_budget": total_budget,
                "confidence_percentile": confidence_percentile,
            })
            result = self._deepthink_online(
                prompt, output, 
                warmup_traces, total_budget, confidence_percentile,
                window_size, sampling_params
            )
        else:
            output.config.update({
                "budget": budget,
            })
            result = self._deepthink_offline(
                prompt, output,
                budget, window_size, sampling_params
            )
        
        # Perform multiple voting analysis if requested
        if compute_multiple_voting and output.all_traces:
            print("Computing multiple voting results...")
            voting_start = time.time()
            if output.mode == "online":
                voting_pool = output.all_voting_traces
            else:
                voting_pool = output.all_traces
            output.voting_results = compute_all_voting_results(voting_pool)
            # The three ExploreExploitThink read-outs over the same pool
            # DeepConf votes on, so both methods are scored by identical
            # rules (see deepconfTesting.eet_voting).
            output.voting_results.update(eet_voting_results(voting_pool))
            
            # Set the primary answer to the majority vote result
            if 'majority' in output.voting_results and output.voting_results['majority']:
                output.voted_answer = output.voting_results['majority']['answer']
                output.final_answer = output.voted_answer
            
            voting_time = time.time() - voting_start
            print(f"Multiple voting computed in {voting_time:.2f} seconds")
        
        output.total_time = time.time() - total_start_time
        output.print_summary()
        
        if compute_multiple_voting and output.voting_results:
            output.print_detailed_voting_results()
        
        return output
    
    def _deepthink_online(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        total_budget: int,
        confidence_percentile: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Online deep thinking with confidence-based early stopping"""
        
        processing_start = time.time()
        # The ASC replay's clock: one monotonic frame for every stamp.
        processing_start_perf = time.perf_counter()
        
        # Warmup phase
        print(f"Starting warmup phase...", sampling_params)
        warmup_gen_start = time.time()
        

        # Generate warmup traces
        warmup_params_list = []
        base_seed = time.time_ns()
        for param_id in range(warmup_traces):
            warmup_params = copy.deepcopy(sampling_params)
            warmup_params.n = 1
            warmup_params.seed = base_seed + param_id
            warmup_params_list.append(warmup_params)
        warmup_outputs = self.generate_observed([prompt for _ in range(warmup_traces)], warmup_params_list)
        output.warmup_gen_time = time.time() - warmup_gen_start
        
        # Process warmup results
        warmup_process_start = time.time()
        warmup_result = process_batch_results(warmup_outputs, window_size)
        output.warmup_process_time = time.time() - warmup_process_start
        
        print('Warmup min_confs:', warmup_result['min_confs'])
        output.conf_bar = float(np.percentile(warmup_result['min_confs'],100 - confidence_percentile))
        output.warmup_min_confs = warmup_result['min_confs']
        
        output.warmup_traces = warmup_result['traces']
        output.warmup_tokens = warmup_result['total_tokens']
        
        print(f"Warmup completed: conf_bar={output.conf_bar:.3f}")
        # Warm-up voters become eligible only now, once the threshold they
        # are tested against exists; they are admitted together at the
        # calibration moment, in native order.
        calibration_offset = time.perf_counter() - processing_start_perf
        admissions = [
            {"index": i, "round": 1, "time_s": calibration_offset,
             "tokens": output.warmup_tokens}
            for i in range(warmup_traces)
        ]
        checkpoints = [{"round": 1, "time_s": calibration_offset,
                        "tokens": output.warmup_tokens,
                        "n_admitted": warmup_traces}]
        
        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()
        # final_params = copy.deepcopy(sampling_params)
        # final_params.seed = int(time.time())
        # final_params.n = total_budget - warmup_traces

        final_params_list = []
        for param_id in range(total_budget - warmup_traces):
            final_params = copy.deepcopy(sampling_params)
            final_params.n = 1
            final_params.seed = base_seed + param_id + warmup_traces
            final_params.extra_args = {
                "conf_threshold": output.conf_bar,
                "eos_token_id": self.tokenizer.eos_token_id,
                "conf_group_size": window_size,
                "conf_topk": DEEPCONF_TOPK,
            }
            final_params_list.append(final_params)
        final_offset = time.perf_counter() - processing_start_perf
        final_outputs = self.generate_observed([prompt for _ in range(total_budget - warmup_traces)], final_params_list)
        output.final_gen_time = time.time() - final_gen_start
        # Final-phase voters are eligible the moment they finish unkilled.
        admissions += [
            {"index": warmup_traces + i, "round": 2,
             "time_s": final_offset + req.outputs[0].finish_time_s,
             "tokens": output.warmup_tokens + req.outputs[0].batch_tokens_at_finish}
            for i, req in enumerate(final_outputs)
        ]
        
        # Process final results
        final_process_start = time.time()
        final_result = process_batch_results(final_outputs, window_size)
        output.final_process_time = time.time() - final_process_start
        
        print('Final min_confs:', final_result['min_confs'])
        output.final_min_confs = final_result['min_confs']
        
        output.final_traces = final_result['traces']
        output.final_tokens = final_result['total_tokens']
        
        # Apply confidence threshold to final traces
        for trace in output.final_traces:
            if trace["min_conf"] < output.conf_bar:
                trace["stop_reason"] = "gconf_threshold"
        
        # Combine all traces
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        # Recorded-only ASC replay over DeepConf's own eligible voters:
        # a trace votes only when it cleared conf_bar and answered.
        eligible = [t.get('min_conf', 0) >= output.conf_bar and bool(t.get('extracted_answer'))
                    for t in output.all_traces]
        checkpoints.append({"round": 2, "time_s": time.perf_counter() - processing_start_perf,
                            "tokens": output.total_tokens,
                            "n_admitted": len(admissions)})
        output.asc = shadow_record(
            admissions, checkpoints,
            [t['extracted_answer'] if ok else None for t, ok in zip(output.all_traces, eligible)],
            [t.get('readout_tail_conf') for t in output.all_traces],
            full_time_s=time.perf_counter() - processing_start_perf,
            full_tokens=output.total_tokens,
        )
        return output
    
    def _deepthink_offline(
        self,
        prompt: str,
        output: DeepThinkOutput,
        budget: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Offline deep thinking - generate all traces at once"""
        
        sampling_params_list = []
        base_seed = time.time_ns()
        for param_id in range(budget):
            sampling_params_x = copy.deepcopy(sampling_params)
            sampling_params_x.n = 1
            sampling_params_x.seed = base_seed + param_id
            sampling_params_list.append(sampling_params_x)

        # Generate all traces at once
        print(f"Generating {budget} traces...", sampling_params)
        generation_start = time.time()
        vllm_outputs = self.llm.generate([prompt for _ in range(budget)], sampling_params_list)
        output.generation_time = time.time() - generation_start
        
        # Process results
        processing_start = time.time()
        processed_results = process_batch_results_offline(vllm_outputs, window_size)
        
        output.all_traces = processed_results['traces']
        output.total_tokens = processed_results['total_tokens']
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _perform_basic_voting(self, output: DeepThinkOutput):
        """Perform basic weighted majority voting (for backward compatibility)"""
        voting_answers = []
        voting_weights = []
        
        if output.mode == "online":
            output.all_voting_traces = []
            # Add warmup traces above threshold
            for trace in output.warmup_traces:
                if trace.get('min_conf', 0) >= output.conf_bar and trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
                    output.all_voting_traces.append(trace)
            
            # Add final traces (skip early stopped ones)
            for trace in output.final_traces:
                if trace.get('stop_reason') == 'gconf_threshold':
                    continue
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
                    output.all_voting_traces.append(trace)
        else:
            # Offline mode - use all traces with valid answers
            for trace in output.all_traces:
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(1.0)
        
        output.voting_answers = voting_answers
        output.voting_weights = voting_weights
        
        # Get voted answer (basic method)
        output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        output.final_answer = output.voted_answer
        
        # Calculate token statistics
        if output.mode == "online":
            output.avg_tokens_per_warmup_trace = output.warmup_tokens / len(output.warmup_traces) if output.warmup_traces else 0
            output.avg_tokens_per_final_trace = output.final_tokens / len(output.final_traces) if output.final_traces else 0
        
        print(f'Basic voting candidates: {len(voting_answers)}')
        if voting_answers:
            print(f'Sample voting answers: {voting_answers[:5]}')