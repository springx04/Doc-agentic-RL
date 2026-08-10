"""BayesTool-RL local runtime.

The package is intentionally dependency-light at its boundary.  The rollout
integration can import the world and decision runtimes without importing a
model server, while the optional neural filter and Q head use PyTorch when it
is available.
"""

from .config import BayesToolConfig, default_config, stage_definition, stage_schedule
from .schema import (
    BeliefSnapshot,
    BranchRecord,
    TaskStateView,
    ToolQualityPosterior,
    ToolQualitySpec,
    ToolWorldSpec,
    WorldEvent,
)
from .world import (
    CleanResultCache,
    DEFAULT_TOOL_ARGUMENT_CAPABILITIES,
    ObservationCorruptionAdapter,
    WorldSamplingContext,
    WorldRuntime,
    document_hash,
    sample_session_state,
    stable_seed,
    tool_world_spec_from_dict,
)
from .replay import (
    CANONICAL_REPLAY_SCHEMA_VERSION,
    canonical_replay_is_valid,
    export_canonical_replay,
    validate_canonical_replay,
)
from .belief import (
    FEATURE_NAMES,
    BeliefRuntime,
    ObservationFeatures,
    ToolWorldFilterNetwork,
    extract_observation_features,
)
from .decision import DecisionController, BayesQHead, q_feature_vectors
from .grouping import (
    ALLOWED_GROUP_SIZES,
    DEFAULT_POLICY_VERSION,
    WORLD_SLOT_ROLES,
    QuestionRolloutPlan,
    RealizationPlan,
    compute_hierarchical_loss_weights,
    make_question_rollout_plan,
    make_runtime_state_digest,
    validate_bayestool_question_records,
    validate_question_rollout_plan_records,
)

__all__ = [
    "BayesToolConfig",
    "default_config",
    "stage_definition",
    "stage_schedule",
    "BeliefSnapshot",
    "BranchRecord",
    "TaskStateView",
    "ToolQualityPosterior",
    "ToolQualitySpec",
    "ToolWorldSpec",
    "WorldEvent",
    "CleanResultCache",
    "WorldSamplingContext",
    "DEFAULT_TOOL_ARGUMENT_CAPABILITIES",
    "ObservationCorruptionAdapter",
    "WorldRuntime",
    "document_hash",
    "stable_seed",
    "sample_session_state",
    "tool_world_spec_from_dict",
    "FEATURE_NAMES",
    "BeliefRuntime",
    "ObservationFeatures",
    "ToolWorldFilterNetwork",
    "extract_observation_features",
    "DecisionController",
    "BayesQHead",
    "q_feature_vectors",
    "WORLD_SLOT_ROLES",
    "ALLOWED_GROUP_SIZES",
    "DEFAULT_POLICY_VERSION",
    "QuestionRolloutPlan",
    "RealizationPlan",
    "make_question_rollout_plan",
    "make_runtime_state_digest",
    "validate_bayestool_question_records",
    "validate_question_rollout_plan_records",
    "compute_hierarchical_loss_weights",
    "CANONICAL_REPLAY_SCHEMA_VERSION",
    "export_canonical_replay",
    "validate_canonical_replay",
    "canonical_replay_is_valid",
]
