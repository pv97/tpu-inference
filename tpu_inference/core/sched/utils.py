# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def update_vllm_scheduler_for_exporting_expert_ids():
    # Monkeypatch vLLM Scheduler to attach expert indices to outputs
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import \
        RoutedExpertsReader
    from vllm.v1.core.sched.scheduler import Scheduler

    class DummyRoutedExpertsReader:

        @staticmethod
        def create():
            return DummyRoutedExpertsReader()

        def attach_buffer(self, *args, **kwargs):
            pass

        def get_routed_experts(self, *args, **kwargs):
            return None

    # Since we are reusing the upstream enable_return_routed_experts flag,
    # we need to stub out the actual RoutedExpertsReader class which the
    # upstream scheduler tries to create.
    RoutedExpertsReader.create = DummyRoutedExpertsReader.create

    original_update_from_output = Scheduler.update_from_output

    def custom_update_from_output(self, scheduler_output, model_runner_output):
        expert_indices = getattr(model_runner_output, "expert_indices", None)

        if expert_indices is not None:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            current_token_offset = 0
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                start_idx = current_token_offset
                end_idx = start_idx + num_tokens_scheduled
                current_token_offset = end_idx

                request = self.requests.get(req_id)
                if request is not None:
                    step_experts = expert_indices[:, start_idx:
                                                  end_idx, :].transpose(
                                                      1, 0, 2)
                    if not hasattr(request, "_accumulated_routed_experts"):
                        request._accumulated_routed_experts = []
                    request._accumulated_routed_experts.append(step_experts)

        return original_update_from_output(self, scheduler_output,
                                           model_runner_output)

    Scheduler.update_from_output = custom_update_from_output

    original_get_routed_experts = Scheduler._get_routed_experts

    def custom_get_routed_experts(self, request):
        if hasattr(request, "_accumulated_routed_experts"
                   ) and request._accumulated_routed_experts:
            import numpy as np
            return np.concatenate(request._accumulated_routed_experts, axis=0)
        return original_get_routed_experts(self, request)

    Scheduler._get_routed_experts = custom_get_routed_experts


def patch_vllm_scheduler_for_continue_decode():
    # Monkeypatch vLLM Scheduler to support continue decode multi-step scheduling
    from vllm.v1.core.sched.scheduler import Scheduler

    # Avoid patching multiple times
    if not getattr(Scheduler, "_continue_decode_patched", False):
        original_schedule = Scheduler.schedule

        def patched_schedule(scheduler_self):
            enable_continue_decode = scheduler_self.vllm_config.additional_config.get(
                "enable_continue_decode", False)
            is_pooling_model = scheduler_self.vllm_config.model_config.runner_type == "pooling"

            if enable_continue_decode and not is_pooling_model:
                user_max_decode_steps = scheduler_self.vllm_config.additional_config.get(
                    "max_decode_steps", 10)

                # Unconditionally fake placeholders for all active decode requests in running.
                # vLLM's scheduler and speculative rollback natively handle all mixed-batch fallbacks!
                for request in scheduler_self.running:
                    is_request_decode = request.num_computed_tokens >= request.num_prompt_tokens
                    if is_request_decode:
                        if user_max_decode_steps > 1:
                            request.spec_token_ids = [-1] * (
                                user_max_decode_steps - 1)
                        else:
                            request.spec_token_ids = []

            # Call original schedule (natively handles block allocation & placeholder tracking)
            scheduler_output = original_schedule(scheduler_self)

            # Clear the fake spec_token_ids immediately to avoid side effects
            for request in scheduler_self.running:
                request.spec_token_ids = []

            return scheduler_output

        # Stub out make_spec_decoding_stats to prevent speculative metrics assertion crashes
        def patched_make_stats(scheduler_self, *args, **kwargs):
            return None

        Scheduler.schedule = patched_schedule
        Scheduler.make_spec_decoding_stats = patched_make_stats
        Scheduler._continue_decode_patched = True
