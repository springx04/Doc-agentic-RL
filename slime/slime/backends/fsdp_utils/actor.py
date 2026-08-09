import logging
import os
import random
from argparse import Namespace
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig

from slime.ray.train_actor import TrainRayActor
from slime.utils import logging_utils, train_dump_utils, train_metric_utils
from slime.utils.action_training import apply_action_training_overrides, build_action_training_overrides
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import Box
from slime.utils.ppo_utils import compute_approx_kl, compute_gspo_kl, compute_opsm_mask, compute_policy_loss
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.profile_utils import TrainProfiler
from slime.utils.timer import Timer, inverse_timer, timer, with_defer

from . import checkpoint
from .data_packing import pack_sequences, unpack_sequences
from .lr_scheduler import get_lr_scheduler
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        # Setup device mesh for data parallelism
        self._setup_device_mesh()
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self._enable_true_on_policy_optimizations(args)
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        init_context = self._get_init_weight_context_manager()

        with init_context():
            model = self.get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        model.train()

        # Apply LoRA adapters before FSDP wrapping
        self._is_lora = getattr(self.args, "use_lora", False)
        if self._is_lora:
            from .lora_utils import apply_lora, propagate_no_split_modules

            model = apply_lora(model, self.args)
            model = propagate_no_split_modules(model)

        full_state = model.state_dict()

        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

        self.model = model

        if args.gradient_checkpointing:
            # LoRA freezes base params, so reentrant checkpointing fails
            # (inputs don't have requires_grad=True). Use non-reentrant mode.
            gc_kwargs = {"use_reentrant": False} if self._is_lora else {}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        # For LoRA, only pass trainable parameters to the optimizer
        optim_params = (
            [p for p in self.model.parameters() if p.requires_grad]
            if self._is_lora
            else self.model.parameters()
        )

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                optim_params,
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        # When LoRA is enabled, PEFT wraps parameter names with prefixes that
        # SGLang doesn't understand.  Strip them and skip adapter-only params.
        if self._is_lora:

            def _lora_name_transform(name: str) -> str | None:
                # Skip LoRA adapter parameters — SGLang only needs merged base weights
                if "lora_A" in name or "lora_B" in name or "lora_embedding" in name:
                    return None
                # Strip PEFT outer prefix: base_model.model.xxx → xxx
                if name.startswith("base_model.model."):
                    name = name[len("base_model.model."):]
                # Strip .base_layer. inserted by PEFT for LoRA-targeted modules
                name = name.replace(".base_layer.", ".")
                return name

            self.weight_updater._name_transform = _lora_name_transform

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _enable_true_on_policy_optimizations(self, args):
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            from .models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    def _setup_device_mesh(self) -> None:
        """Setup device mesh for data parallelism."""
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Pure data parallelism
        self.dp_size = world_size
        self.dp_rank = rank

        # Create 1D device mesh for data parallelism
        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh

        logger.info(f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}")

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict[str, torch.Tensor]],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities for a list of packed batches.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            packed_batches: A list of packed batch dictionaries produced by
                `pack_sequences`, each containing at least `tokens` and
                `position_ids`; may also include multimodal keys like `pixel_values`.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs". The
            actual per-sequence results are written in-place into each element of
            `packed_batches` under the same key and can be read back by callers.

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

            active_model = self.ref_model
            active_model.eval()
        else:
            active_model = self.model

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    log_probs_result, entropy_result = self._forward_log_probs(
                        active_model,
                        batch,
                        compute_entropy=self.args.entropy_coef != 0.0,
                    )
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if store_prefix == "":
                        batch["entropy"] = entropy_result
            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    def _packed_data(
        self, rollout_data: dict[str, list[torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        """Pack variable-length sequences for efficient processing.

        Parameters:
            rollout_data: Dictionary of lists containing sequence-level tensors
                such as `tokens`, `loss_masks`, `rewards`, `response_lengths`,
                `advantages`, `returns`, and optional `rollout_log_probs`.

        Returns:
            A pair `(packed_batches, grad_accum)` where `packed_batches` is a list
            of packed batch dictionaries and `grad_accum` lists the micro-batch
            indices at which to perform optimizer steps.
        """
        # Pack sequences efficiently
        tokens = rollout_data["tokens"]

        packed_batches = []
        mbs_size_list = []
        local_batch_size = self.args.global_batch_size // self.dp_size
        assert (
            self.args.global_batch_size % self.dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {self.dp_size}"
        # Use global_batch_size for splitting when max_tokens_per_gpu is enabled
        if self.args.use_dynamic_batch_size:
            max_tokens = self.args.max_tokens_per_gpu

            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
                len(tokens) // local_batch_size
            )

        start = 0
        for mbs_size in num_microbatches:
            end = start + local_batch_size
            packed_batches.extend(
                pack_sequences(
                    rollout_data["tokens"][start:end],
                    rollout_data["loss_masks"][start:end],
                    rollout_data["rewards"][start:end],
                    rollout_data["raw_reward"][start:end],
                    rollout_data["response_lengths"][start:end],
                    rollout_data["advantages"][start:end],
                    rollout_data["returns"][start:end],
                    rollout_log_probs=(
                        rollout_data["rollout_log_probs"][start:end] if "rollout_log_probs" in rollout_data else None
                    ),
                    multimodal_train_inputs=(
                        rollout_data["multimodal_train_inputs"][start:end]
                        if "multimodal_train_inputs" in rollout_data
                        else None
                    ),
                    num_packs=mbs_size,
                )
            )
            start = end
        grad_accum = list(accumulate(num_microbatches))

        return packed_batches, grad_accum

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: A Box handle wrapping a Ray object reference to a
                dictionary with rollout tensors and metadata (e.g., `tokens`,
                `loss_masks`, `rewards`, `response_lengths`, optional
                `rollout_log_probs`, etc.). It will be fetched and partitioned
                by `process_rollout_data` based on data-parallel rank/size.
        """
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _log_rollout_data(self, rollout_id: int, rollout_data, packed_batches):
        log_dict = {}
        if "raw_reward" in rollout_data and dist.get_rank() == 0:
            raw_reward_list = rollout_data["raw_reward"]
            if raw_reward_list:
                log_dict["rollout/raw_reward"] = sum(raw_reward_list) / len(raw_reward_list)

        for metric_key in ["log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"]:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for _mbs_id, batches in enumerate(packed_batches):
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            logger.info(f"rollout {rollout_id}: {log_dict}")
            log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            logging_utils.log(self.args, log_dict, step_key="rollout/step")

        if self.args.ci_test and self.args.true_on_policy_mode:
            assert log_dict["rollout/log_probs"] == log_dict["rollout/rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['rollout/log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout/rollout_log_probs']})"
            )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        if self.args.advantage_estimator == "bayes_grpo":
            # Bayes-ARPO uses the raw complete-trajectory utility and a
            # weighted sibling baseline.  The FSDP path does not call the
            # Megatron advantage helper, so compute the same estimator here
            # before packing the response-space tensors.
            raw_rewards = list(rollout_data.get("raw_reward", rollout_data["rewards"]))
            local_count = len(rollout_data["response_lengths"])
            partition = rollout_data.get("_partition")
            if len(raw_rewards) != local_count:
                if partition is not None and len(partition) == local_count:
                    raw_rewards = [raw_rewards[int(index)] for index in partition]
                else:
                    raw_rewards = list(rollout_data["rewards"])
            # _packed_data expects raw_reward to be aligned with the local
            # partition, while the rollout splitter intentionally preserves
            # the global raw-reward list for other backends.
            rollout_data["raw_reward"] = raw_rewards
            group_ids = (
                rollout_data.get("bayes_group_ids")
                or rollout_data.get("sibling_group_ids")
                or rollout_data.get("group_indices")
            )
            if group_ids is None or len(group_ids) != len(raw_rewards):
                group_ids = list(range(len(raw_rewards)))
            sibling_weights = rollout_data.get("bayes_sibling_weights") or [1.0] * len(raw_rewards)
            if len(sibling_weights) != len(raw_rewards):
                sibling_weights = [1.0] * len(raw_rewards)

            grouped: dict[str, list[int]] = {}
            for index, group_id in enumerate(group_ids):
                grouped.setdefault(str(group_id), []).append(index)

            baselines = [0.0] * len(raw_rewards)
            sibling_advantages = [0.0] * len(raw_rewards)
            for indices in grouped.values():
                weights = [max(0.0, float(sibling_weights[index])) for index in indices]
                denominator = sum(weights) or float(len(indices))
                baseline = sum(
                    weight * float(raw_rewards[index])
                    for index, weight in zip(indices, weights, strict=True)
                ) / denominator
                for index in indices:
                    baselines[index] = baseline
                    sibling_advantages[index] = float(raw_rewards[index]) - baseline

            advantages = []
            effective_loss_masks = []
            consumed_action_indices = []
            action_rewards = rollout_data.get("action_rewards", [])
            action_token_spans = rollout_data.get("action_token_spans", [])
            for i, sequence_advantage_value in enumerate(sibling_advantages):
                response_length = int(rollout_data["response_lengths"][i])
                sequence_advantage = torch.tensor(
                    [sequence_advantage_value] * response_length,
                    dtype=torch.float32,
                )
                loss_mask = list(rollout_data["loss_masks"][i])
                signal = apply_action_training_overrides(
                    response_length=response_length,
                    advantages=sequence_advantage,
                    returns=sequence_advantage,
                    loss_mask=loss_mask,
                    assistant_token_masks=action_token_spans[i] if i < len(action_token_spans) else [],
                    action_rewards=action_rewards[i] if i < len(action_rewards) else [],
                )
                advantages.append(signal["advantages"])
                effective_loss_masks.append(signal["loss_mask"])
                consumed_action_indices.append(signal["consumed_action_indices"])

            rollout_data["loss_masks"] = effective_loss_masks
            rollout_data["advantages"] = rollout_data["returns"] = advantages
            rollout_data["action_reward_consumed"] = consumed_action_indices
            rollout_data["bayes_sibling_baselines"] = baselines
            rollout_data["bayes_sibling_advantages"] = sibling_advantages
        elif self.args.advantage_estimator in ["grpo", "gspo"]:
            advantages = []
            effective_loss_masks = []
            consumed_action_indices = []
            action_rewards = rollout_data.get("action_rewards", [])
            action_token_spans = rollout_data.get("action_token_spans", [])
            for i, reward in enumerate(rollout_data["rewards"]):
                response_length = int(rollout_data["response_lengths"][i])
                sequence_advantage = torch.tensor(
                    [reward] * response_length,
                    dtype=torch.float32,
                )
                loss_mask = list(rollout_data["loss_masks"][i])
                signal = build_action_training_overrides(
                    response_length,
                    action_token_spans[i] if i < len(action_token_spans) else [],
                    action_rewards[i] if i < len(action_rewards) else [],
                )
                action_advantages = signal["action_advantages"]
                action_mask = signal["action_token_mask"]
                for position, enabled in enumerate(action_mask):
                    if enabled:
                        # Rejected action tokens receive the negative action
                        # advantage, never the positive terminal reward.
                        sequence_advantage[position] = float(action_advantages[position])
                        loss_mask[position] = 1
                advantages.append(sequence_advantage)
                effective_loss_masks.append(loss_mask)
                consumed_action_indices.append(signal["consumed_action_indices"])
            rollout_data["loss_masks"] = effective_loss_masks
            rollout_data["advantages"] = rollout_data["returns"] = advantages
            rollout_data["action_reward_consumed"] = consumed_action_indices
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        if dist.get_rank() == 0:
            sequence_lengths = sorted(len(sequence) for sequence in rollout_data["tokens"])
            p95_index = min(len(sequence_lengths) - 1, max(0, int(len(sequence_lengths) * 0.95) - 1))
            multimodal_count = sum(
                1
                for value in rollout_data.get("multimodal_train_inputs", [])
                if isinstance(value, dict) and value
            )
            logger.info(
                "TRAIN_ROLLOUT_LENGTHS rollout_id=%s samples=%s max_tokens=%s p95_tokens=%s "
                "total_tokens=%s multimodal_samples=%s",
                rollout_id,
                len(sequence_lengths),
                max(sequence_lengths, default=0),
                sequence_lengths[p95_index] if sequence_lengths else 0,
                sum(sequence_lengths),
                multimodal_count,
            )

        packed_batches, grad_accum = self._packed_data(rollout_data)

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            self._compute_log_prob("ref", packed_batches, store_prefix="ref_")

        self._compute_log_prob("actor", packed_batches)
        self._log_rollout_data(rollout_id, rollout_data, packed_batches)

        with timer("actor_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in self.prof.iterate_train_actor(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                try:
                    self._train_step(
                        packed_batch=packed_batch,
                        reported_accum=reported_accum,
                        mbs_id=mbs_id,
                        grad_accum=grad_accum,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    try:
                        cu_seqlens = packed_batch.get("cu_seqlens")
                        sequence_lengths = (
                            (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
                            if isinstance(cu_seqlens, torch.Tensor)
                            else []
                        )
                        multimodal_shapes = {
                            key: tuple(value.shape)
                            for key, value in packed_batch.get("multimodal_train_inputs", {}).items()
                            if isinstance(value, torch.Tensor)
                        }
                        free_memory, total_memory = torch.cuda.mem_get_info(torch.cuda.current_device())
                        logger.error(
                            "FSDP_TRAIN_OOM_DIAGNOSTIC rollout_id=%s mbs_id=%s rank=%s "
                            "packed_tokens=%s sequence_lengths=%s multimodal_shapes=%s "
                            "allocated_bytes=%s reserved_bytes=%s free_bytes=%s total_bytes=%s",
                            rollout_id,
                            mbs_id,
                            dist.get_rank(),
                            int(packed_batch.get("tokens").numel()) if isinstance(packed_batch.get("tokens"), torch.Tensor) else None,
                            sequence_lengths,
                            multimodal_shapes,
                            torch.cuda.memory_allocated(),
                            torch.cuda.memory_reserved(),
                            free_memory,
                            total_memory,
                            exc_info=True,
                        )
                    except Exception:
                        logger.exception(
                            "FSDP_TRAIN_OOM_DIAGNOSTIC failed while collecting memory details"
                        )
                    raise

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Update ref model if needed (copy actor weights to ref)
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            if self._is_lora:
                from .lora_utils import get_merged_state_dict

                actor_state = get_merged_state_dict(self.model)
            else:
                actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        # Compute log probs and entropy.  Qwen3-VL can return logits only for
        # the response positions; keeping the multimodal prompt logits would
        # allocate [sequence_length, vocab_size] even though every prompt
        # token has a zero loss mask.
        need_entropy = self.args.entropy_coef != 0.0
        log_probs, entropy_result = self._forward_log_probs(
            self.model,
            packed_batch,
            compute_entropy=need_entropy,
        )
        packed_batch["cur_log_probs"] = log_probs
        packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)

        old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if old_log_prob_key not in batch or not isinstance(batch[old_log_prob_key], torch.Tensor)
        ]
        if missing_old_log_probs:
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"use_rollout_logprobs is set to {self.args.use_rollout_logprobs}. Missing in batches: {missing_old_log_probs}"
            )
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        ppo_kl = old_log_probs - log_probs

        if self.args.use_opsm:
            opsm_mask, opsm_clipfrac = compute_opsm_mask(
                args=self.args,
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                advantages=[batch["advantages"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        if self.args.advantage_estimator == "gspo":
            ppo_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        if self.args.use_opsm:
            pg_loss = pg_loss * opsm_mask

        def _has_rollout_log_probs(batch) -> bool:
            rollout_tensor = batch.get("rollout_log_probs")
            return isinstance(rollout_tensor, torch.Tensor) and rollout_tensor.numel() > 0

        has_rollout_log_probs = all(_has_rollout_log_probs(batch) for batch in unpacked_batches)
        rollout_log_probs = (
            torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
            if has_rollout_log_probs
            else None
        )

        if self.args.calculate_per_token_loss:
            pg_loss = sum_of_token(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_token(ppo_kl.abs(), response_lengths, loss_masks)
        else:
            pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

        # Only compare rollout vs. train log probs when they originate from different stages.
        train_rollout_logprob_abs_diff = None
        if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
            train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
            train_rollout_logprob_abs_diff = sum_of_sample_mean(
                train_rollout_logprob_abs_diff, response_lengths, loss_masks
            ).detach()

        entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
        entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

            loss = loss + self.args.kl_loss_coef * kl_loss

        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if train_rollout_logprob_abs_diff is not None:
            reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

        if self.args.use_kl_loss:
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_opsm:
            reported["opsm_clipfrac"] = opsm_clipfrac

        # Scale loss for gradient accumulation
        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        # Accumulate reported metrics (store tensors for later mean)
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) in grad_accum:
            # TODO: check if the grad norm is global grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            # the grad norm used to be of DTensor
            grad_norm = float(grad_norm)

            self.optimizer.step()
            # Update learning rate
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            reduced_aggregated = [None] * self.dp_size
            dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            reported_accum.clear()
            if dist.get_rank() == 0:
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                log_dict["train/grad_norm"] = grad_norm

                # Log learning rate per parameter group; use scheduler's last computed LRs
                lr_values = self.lr_scheduler.get_last_lr()
                for gid, _group in enumerate(self.optimizer.param_groups):
                    log_dict[f"train/lr-pg_{gid}"] = lr_values[gid]

                kl_info = ""
                if self.args.use_kl_loss and "kl_loss" in aggregated:
                    kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    logger.info(kl_info)
                logger.info(f"step {self.global_step}: {log_dict}")

                log_dict["train/step"] = self.global_step
                logging_utils.log(self.args, log_dict, step_key="train/step")
            self.global_step += 1

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        # Merge LoRA into base weights before syncing to SGLang rollout engines.
        # SGLang doesn't understand LoRA adapters, so it needs the merged model.
        if self._is_lora:
            self.model.merge_adapter()
        try:
            self.weight_updater.update_weights()
        finally:
            if self._is_lora:
                self.model.unmerge_adapter()

        if self.args.ci_test and len(rollout_engines) > 0:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        clear_memory()

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            with init_context():
                ref_model = self.get_model_cls().from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    attn_implementation=self.args.attn_implementation,
                )

            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _supports_selective_logits(self) -> bool:
        """Whether the active VLM exposes arbitrary ``logits_to_keep`` indices."""

        # Qwen3-VL's HF forward supports an index tensor, which is important
        # for packed multimodal sequences.  Do not pass this model-specific
        # kwarg to text-only/older models that may not accept it.
        return hasattr(self.hf_config, "vision_config") and getattr(self.hf_config, "model_type", None) == "qwen3_vl"

    @staticmethod
    def _response_logit_positions(packed_sequence: dict, device: torch.device) -> torch.Tensor:
        """Return flattened positions whose logits predict response tokens."""

        cu_seqlens = packed_sequence["cu_seqlens"].tolist()
        response_lengths = packed_sequence["response_lengths"]
        positions: list[int] = []
        for sequence_index, response_length in enumerate(response_lengths):
            start = int(cu_seqlens[sequence_index])
            end = int(cu_seqlens[sequence_index + 1])
            response_length = int(response_length)
            available = max(0, end - start - 1)
            if response_length < 0 or response_length > available:
                raise ValueError(
                    f"response length {response_length} is incompatible with packed sequence "
                    f"[{start}, {end})"
                )
            # Position p predicts input_ids[p + 1].  The response occupies
            # the final response_length input tokens of this sequence.
            positions.extend(range(end - response_length - 1, end - 1))
        return torch.tensor(positions, dtype=torch.long, device=device)

    def _forward_log_probs(
        self,
        model,
        packed_sequence: dict,
        *,
        compute_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one packed forward while avoiding full VLM vocabulary logits."""

        if self._supports_selective_logits():
            # The model still processes the complete multimodal prefix so
            # causal attention and MRoPE semantics are unchanged.  Only the
            # final response-token projection is retained, which removes the
            # otherwise dominant ``T * vocab_size`` allocation.
            model_device = next(model.parameters()).device
            positions = self._response_logit_positions(packed_sequence, model_device)
            if positions.numel() > 0:
                model_args = self._get_model_inputs_args(packed_sequence, logits_to_keep=positions)
                logits = model(**model_args).logits.squeeze(0)
                target_positions = positions + 1
                target_tokens = packed_sequence["tokens"].to(device=model_device).index_select(0, target_positions)
                selected_log_probs, selected_entropy = get_selected_logprob_and_entropy(
                    logits=logits,
                    target_tokens=target_tokens,
                    allow_compile=not self.args.true_on_policy_mode,
                    temperature=self.args.rollout_temperature,
                    compute_entropy=compute_entropy,
                )
                # Preserve the existing packed/unpack interface: non-response
                # positions are zero but never contribute because their masks
                # are zero.  The vector is tiny compared with the logits.
                packed_length = max(0, int(packed_sequence["tokens"].numel()) - 1)
                log_probs = torch.zeros(packed_length, dtype=torch.float32, device=model_device)
                entropy = torch.zeros_like(log_probs)
                log_probs.index_copy_(0, positions, selected_log_probs.float())
                entropy.index_copy_(0, positions, selected_entropy.float())
                return log_probs, entropy

        model_args = self._get_model_inputs_args(packed_sequence)
        logits = model(**model_args).logits.squeeze(0)
        return get_logprob_and_entropy(
            logits=logits,
            target_tokens=packed_sequence["tokens"],
            allow_compile=not self.args.true_on_policy_mode,
            temperature=self.args.rollout_temperature,
            compute_entropy=compute_entropy,
        )

    def _get_model_inputs_args(self, packed_sequence: dict, logits_to_keep: torch.Tensor | None = None) -> dict:
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        position_ids = packed_sequence["position_ids"].unsqueeze(0)

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        if packed_sequence.get("multimodal_train_inputs"):
            model_args.update(packed_sequence["multimodal_train_inputs"])
        if logits_to_keep is not None:
            model_args["logits_to_keep"] = logits_to_keep
        return model_args


def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.

    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.

    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.

    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    # Computing the full ``log_softmax`` tensor is needlessly expensive for a
    # policy-gradient batch: only one vocabulary entry per token is used.  The
    # equivalent gather/log-normalizer formulation keeps the autograd graph
    # over the original logits without materialising another [T, V] tensor.
    target_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    log_normalizer = torch.logsumexp(logits, dim=-1)
    return target_logits - log_normalizer


selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


def gather_log_probs_packed(
    shifted_logits: torch.Tensor,
    input_ids: torch.Tensor,
    allow_compile: bool,
    cu_seqlens: torch.Tensor | float | None = None,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather next-token log probabilities for packed sequences.

    Parameters:
        logits: Model logits of shape [B, T, V] or [T, V].
        input_ids: Token ids of shape [B, T] or [T].
        cu_seqlens: Optional cumulative sequence lengths (unused here). Present
            for API compatibility with callers.

    Returns:
        A tensor of shape [T-1] (or [B, T-1]) with log-probabilities of targets.
    """
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    if shifted_logits.dim() == 3:
        # Remove batch dimension for packed sequences
        shifted_logits = shifted_logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        shifted_logits = shifted_logits.div(temperature)

    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


def get_logprob_and_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy.

    Parameters:
        logits: Model output logits with shape [seq_len, vocab_size]
        target_tokens: Target tokens with shape [seq_len]
        allow_compile: Whether to allow compilation
        temperature: Temperature parameter (optional)
        compute_entropy: If False, skip the expensive full-vocab softmax and
            return zeros for entropy. Saves ~3x [seq_len, vocab_size] memory.

    Returns:
        log_probs: Log probabilities with shape [seq_len - 1]
        entropy: Entropy with shape [seq_len - 1]
    """
    shifted_logits = logits[:-1, :]
    log_probs = gather_log_probs_packed(
        shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
    )
    if compute_entropy:
        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
        probs = torch.softmax(shifted_logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
    else:
        entropy = torch.zeros_like(log_probs)
    return log_probs, entropy


def get_selected_logprob_and_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probs for a tensor of already-selected response positions."""

    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if temperature is not None:
        logits = logits.div(temperature)

    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    log_probs = selective_log_softmax(logits, target_tokens.to(device=logits.device))
    if compute_entropy:
        log_probs_full = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
    else:
        entropy = torch.zeros_like(log_probs)
    return log_probs, entropy


def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Compute sum of per-sample means across variable-length responses.

    Parameters:
        x: Flat tensor containing concatenated per-token values across samples.
        response_lengths: Lengths of each sample's response segment in `x`.
        loss_masks: Per-sample masks aligned with `response_lengths`.

    Returns:
        A scalar tensor equal to the sum over samples of the mean value within
        each sample's response segment.
    """
    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = getattr(model, "_no_split_modules", None)
    # When PEFT wraps the model, _no_split_modules may live on the inner model
    if not layer_cls_to_wrap and hasattr(model, "base_model"):
        inner = getattr(model.base_model, "model", model.base_model)
        layer_cls_to_wrap = getattr(inner, "_no_split_modules", None)
    assert layer_cls_to_wrap and len(layer_cls_to_wrap) > 0 and next(iter(layer_cls_to_wrap)) is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # Determine precision policy based on args
    param_dtype = torch.bfloat16  # Default to bf16 as before
    reduce_dtype = torch.float32

    if args.fp16:
        param_dtype = torch.float16

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def sum_of_token(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    return sum(
        [
            (x_i * loss_mask_i).sum()
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )
