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


def patch_vllm_scheduler_for_continue_decode(vllm_config=None):
    # Monkeypatch vLLM Scheduler to support continue decode multi-step scheduling
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, RequestStatus

    from tpu_inference.logger import init_logger

    logger = init_logger(__name__)

    # Avoid patching multiple times
    if not getattr(Scheduler, "_continue_decode_patched", False):
        max_decode_steps = 1
        if vllm_config:
            additional_config = vllm_config.additional_config
            if additional_config and additional_config.get(
                    "enable_continue_decode", False):
                max_decode_steps = additional_config.get(
                    "max_decode_steps", 10)

        # Store max_decode_steps on Request class so it can be accessed in property
        Request.max_continue_decode_steps = max_decode_steps

        # 1. Patch Request.num_tokens_with_spec property

        @property
        def patched_num_tokens_with_spec(self):
            # If running and in decode phase (num_computed_tokens >= num_prompt_tokens)
            if self.status == RequestStatus.RUNNING and self.num_computed_tokens >= self.num_prompt_tokens:
                max_steps = getattr(Request, "max_continue_decode_steps", 1)
                max_tokens_limit = self.num_prompt_tokens + self.max_tokens
                return min(self.num_computed_tokens + max_steps,
                           max_tokens_limit)
            else:
                return len(self._all_token_ids) + len(self.spec_token_ids)

        Request.num_tokens_with_spec = patched_num_tokens_with_spec

        # 2. Patch Scheduler.update_from_output to perform rollback on mismatch
        original_update_from_output = Scheduler.update_from_output

        def patched_update_from_output(scheduler_self, scheduler_output,
                                       model_runner_output):
            # Call original first (which appends tokens and marks stopped)
            outputs = original_update_from_output(scheduler_self,
                                                  scheduler_output,
                                                  model_runner_output)

            actual_steps = getattr(model_runner_output, "actual_steps", None)
            if actual_steps is None:
                actual_steps = 1

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                request = scheduler_self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue

                computed_before_schedule = request.num_computed_tokens - num_tokens_scheduled
                if computed_before_schedule >= request.num_prompt_tokens:
                    mismatch = num_tokens_scheduled - actual_steps
                    if mismatch > 0:
                        if request.num_computed_tokens > 0:
                            request.num_computed_tokens -= mismatch
                        if request.num_output_placeholders > 0:
                            request.num_output_placeholders -= mismatch
                            if request.num_output_placeholders < 0:
                                request.num_output_placeholders = 0
                        logger.info(
                            f"[PATCH] update_from_output rollback for {req_id}: "
                            f"num_tokens_scheduled={num_tokens_scheduled}, actual_steps={actual_steps}, "
                            f"mismatch={mismatch} -> num_computed_tokens={request.num_computed_tokens}, "
                            f"num_output_placeholders={request.num_output_placeholders}"
                        )
            return outputs

        Scheduler.update_from_output = patched_update_from_output

        # 3. Patch AsyncScheduler if it exists
        try:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            original_async_update = AsyncScheduler._update_after_schedule

            def patched_async_update(scheduler_self, scheduler_output):
                original_async_update(scheduler_self, scheduler_output)
                additional_config = vllm_config.additional_config if vllm_config else None
                if additional_config and additional_config.get(
                        "enable_continue_decode", False):
                    for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items(
                    ):
                        request = scheduler_self.requests[req_id]
                        if request.is_prefill_chunk:
                            continue
                        diff = num_scheduled_token - 1
                        if diff > 0:
                            request.num_output_placeholders += diff

            AsyncScheduler._update_after_schedule = patched_async_update
        except ImportError:
            pass

        Scheduler._continue_decode_patched = True
