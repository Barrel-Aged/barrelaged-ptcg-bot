"""Neural blueprint validation on the exactly enumerable Phase 6B game.

Two deliberately small candidates are trained and compared:

* ``policy`` distills the analytic equilibrium distribution directly from
  public-belief/information-set features;
* ``regret`` approximates a positive regret-matching representation of the
  tabular CFR average strategy, a small inference analogue of a neural CFR
  family (not a claim that the trainer itself is Deep CFR).

Both also learn conditional continuation values for each possible opponent
private strength. Exact best-response enumeration supplies NashConv and
exploitability after training. Success here proves only that a restorable neural
provider can reproduce the microgame reference; it is not a full Pokemon model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .exact_game import (
    DuelCard,
    ExactProfileMetrics,
    ExactTabularBlueprintProvider,
    RocketPrizeDuel,
    TabularStrategyProfile,
    exact_profile_metrics,
)
from .game_theory import (
    ActionDistribution,
    BlueprintCheckpoint,
    CounterfactualValues,
    InformationSet,
    PublicBeliefState,
)


NEURAL_BLUEPRINT_SCHEMA_VERSION = 1
GLOBAL_DUEL_ACTIONS = ("hold", "push", "concede", "contest")
DUEL_BLUEPRINT_FEATURE_NAMES = (
    "player_0",
    "player_1",
    "own_mimikyu",
    "own_spidops",
    "own_mewtwo",
    "history_root",
    "history_hold",
    "history_push",
    "history_hold_push",
    "range_mimikyu",
    "range_spidops",
    "range_mewtwo",
    "pot_commitment",
    "push_count",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


class DuelInformationEncoder:
    feature_names = DUEL_BLUEPRINT_FEATURE_NAMES

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": NEURAL_BLUEPRINT_SCHEMA_VERSION,
                "feature_names": self.feature_names,
                "global_actions": GLOBAL_DUEL_ACTIONS,
            }
        )

    def encode(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> tuple[float, ...]:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("information set/public belief fingerprint mismatch")
        player_text, own_card, history = information_set.info_set_id.split(":", 2)
        player = int(player_text.removeprefix("p"))
        if player not in (0, 1):
            raise ValueError("duel information set has invalid player")
        cards = [card.label for card in DuelCard]
        histories = ("root", "hold", "push", "hold-push")
        if own_card not in cards or history not in histories:
            raise ValueError("duel information set has unknown card/history")
        ranges = {card: 0.0 for card in cards}
        for hypothesis in belief.opponent_range.hypotheses:
            prefix, card = hypothesis.hypothesis_id.split(":", 1)
            if prefix != "opponent" or card not in ranges:
                raise ValueError("duel opponent range hypothesis is invalid")
            ranges[card] = hypothesis.weight
        public = dict(belief.public_features)
        values = (
            float(player == 0),
            float(player == 1),
            *(float(own_card == card) for card in cards),
            *(float(history == item) for item in histories),
            *(ranges[card] for card in cards),
            float(public.get("pot_commitment", 0.0)) / 2.0,
            float(public.get("push_count", 0.0)),
        )
        if len(values) != len(self.feature_names) or not all(map(math.isfinite, values)):
            raise ValueError("duel blueprint feature encoding is invalid")
        return tuple(values)

    @staticmethod
    def legal_action_mask(information_set: InformationSet) -> tuple[bool, ...]:
        legal = {action.action_id for action in information_set.legal_actions}
        if not legal or not legal.issubset(GLOBAL_DUEL_ACTIONS):
            raise ValueError("duel information set has invalid legal action support")
        return tuple(action in legal for action in GLOBAL_DUEL_ACTIONS)


class DuelBlueprintNetwork(nn.Module):
    def __init__(self, *, hidden_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(len(DUEL_BLUEPRINT_FEATURE_NAMES), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.strategy_head = nn.Linear(hidden_dim, len(GLOBAL_DUEL_ACTIONS))
        self.value_head = nn.Linear(hidden_dim, len(DuelCard))
        self.double()

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        return self.strategy_head(hidden), self.value_head(hidden)


@dataclass(frozen=True, slots=True)
class TensorArtifact:
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name or any(size <= 0 for size in self.shape):
            raise ValueError("tensor artifact name/shape is invalid")
        expected = math.prod(self.shape)
        if len(self.values) != expected or not all(math.isfinite(value) for value in self.values):
            raise ValueError("tensor artifact values do not match shape")


@dataclass(frozen=True, slots=True)
class NeuralBlueprintArtifact:
    strategy_mode: str
    hidden_dim: int
    game_fingerprint: str
    deck_fingerprint: str
    encoder_fingerprint: str
    training_dataset_fingerprint: str
    teacher_profile_fingerprint: str
    seed: int
    epochs: int
    learning_rate: float
    final_training_loss: float
    tensors: tuple[TensorArtifact, ...]

    def __post_init__(self) -> None:
        if self.strategy_mode not in {"policy", "regret"}:
            raise ValueError("neural blueprint strategy mode is invalid")
        if self.hidden_dim <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("neural blueprint hyperparameters must be positive")
        if not math.isfinite(self.final_training_loss) or self.final_training_loss < 0:
            raise ValueError("final training loss must be finite and non-negative")
        for field_name in (
            "game_fingerprint",
            "deck_fingerprint",
            "encoder_fingerprint",
            "training_dataset_fingerprint",
            "teacher_profile_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)) or not self.tensors:
            raise ValueError("neural blueprint tensors must be unique and non-empty")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NEURAL_BLUEPRINT_SCHEMA_VERSION,
            "strategy_mode": self.strategy_mode,
            "hidden_dim": self.hidden_dim,
            "game_fingerprint": self.game_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "encoder_fingerprint": self.encoder_fingerprint,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "teacher_profile_fingerprint": self.teacher_profile_fingerprint,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "final_training_loss": self.final_training_loss,
            "feature_names": list(DUEL_BLUEPRINT_FEATURE_NAMES),
            "global_actions": list(GLOBAL_DUEL_ACTIONS),
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NeuralBlueprintArtifact":
        if tuple(raw["feature_names"]) != DUEL_BLUEPRINT_FEATURE_NAMES:
            raise ValueError("neural blueprint feature schema mismatch")
        if tuple(raw["global_actions"]) != GLOBAL_DUEL_ACTIONS:
            raise ValueError("neural blueprint action schema mismatch")
        result = cls(
            strategy_mode=str(raw["strategy_mode"]),
            hidden_dim=int(raw["hidden_dim"]),
            game_fingerprint=str(raw["game_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            encoder_fingerprint=str(raw["encoder_fingerprint"]),
            training_dataset_fingerprint=str(raw["training_dataset_fingerprint"]),
            teacher_profile_fingerprint=str(raw["teacher_profile_fingerprint"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            final_training_loss=float(raw["final_training_loss"]),
            tensors=tuple(
                TensorArtifact(
                    name=str(item["name"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    values=tuple(float(value) for value in item["values"]),
                )
                for item in raw["tensors"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("neural blueprint artifact fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NeuralBlueprintTrainingReport:
    strategy_mode: str
    epochs: int
    seed: int
    final_training_loss: float
    policy_l1_to_teacher: float
    conditional_value_rmse: float
    metrics: ExactProfileMetrics
    artifact_fingerprint: str
    checkpoint_fingerprint: str

    def __post_init__(self) -> None:
        if self.strategy_mode not in {"policy", "regret"} or self.epochs <= 0:
            raise ValueError("neural blueprint report identity is invalid")
        for field_name in (
            "final_training_loss",
            "policy_l1_to_teacher",
            "conditional_value_rmse",
        ):
            value = getattr(self, field_name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        _validate_digest(self.artifact_fingerprint, "artifact_fingerprint")
        _validate_digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")


@dataclass(slots=True)
class NeuralBlueprintTrainingRun:
    provider: "NeuralDuelBlueprintProvider"
    artifact: NeuralBlueprintArtifact
    profile: TabularStrategyProfile
    report: NeuralBlueprintTrainingReport


class NeuralDuelBlueprintProvider:
    """Restorable trained provider for RocketPrizeDuel only."""

    def __init__(
        self,
        game: RocketPrizeDuel,
        network: DuelBlueprintNetwork,
        artifact: NeuralBlueprintArtifact,
    ) -> None:
        if artifact.game_fingerprint != game.fingerprint:
            raise ValueError("neural artifact/game fingerprint mismatch")
        self.game = game
        self.network = network.eval()
        self.artifact = artifact
        self.encoder = DuelInformationEncoder()
        if artifact.encoder_fingerprint != self.encoder.fingerprint:
            raise ValueError("neural artifact/encoder fingerprint mismatch")
        self._checkpoint = BlueprintCheckpoint(
            provider_name=f"neural_duel_{artifact.strategy_mode}_blueprint",
            algorithm=(
                "public_belief_policy_distillation"
                if artifact.strategy_mode == "policy"
                else "neural_regret_matching_distillation_from_tabular_cfr_average"
            ),
            model_fingerprint=artifact.fingerprint,
            game_fingerprint=artifact.game_fingerprint,
            deck_fingerprint=artifact.deck_fingerprint,
            training_steps=artifact.epochs,
            trained=True,
            evaluation_notes=(
                "Trained and exactly evaluated on RocketPrizeDuel only; not an "
                "official Pokemon checkpoint or full-game strength claim."
            ),
        )

    @property
    def checkpoint(self) -> BlueprintCheckpoint:
        return self._checkpoint

    def encode_public_belief(self, belief: PublicBeliefState) -> Sequence[float]:
        public = dict(belief.public_features)
        ranges = {card.label: 0.0 for card in DuelCard}
        for item in belief.opponent_range.hypotheses:
            _, label = item.hypothesis_id.split(":", 1)
            ranges[label] = item.weight
        return (
            *(ranges[card.label] for card in DuelCard),
            float(public.get("pot_commitment", 0.0)) / 2.0,
            float(public.get("push_count", 0.0)),
        )

    def _outputs(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> tuple[Tensor, Tensor]:
        features = torch.tensor(
            self.encoder.encode(information_set, belief), dtype=torch.float64
        ).unsqueeze(0)
        with torch.no_grad():
            strategy, values = self.network(features)
        return strategy.squeeze(0), values.squeeze(0)

    def strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        raw, _ = self._outputs(information_set, belief)
        mask = self.encoder.legal_action_mask(information_set)
        legal_indices = [index for index, legal in enumerate(mask) if legal]
        if self.artifact.strategy_mode == "policy":
            legal_logits = raw[legal_indices]
            probabilities = torch.softmax(legal_logits, dim=0)
        else:
            positive = torch.relu(raw[legal_indices])
            probabilities = (
                positive / positive.sum()
                if float(positive.sum()) > 1e-12
                else torch.full_like(positive, 1.0 / len(legal_indices))
            )
        result = ActionDistribution.normalized(
            {
                GLOBAL_DUEL_ACTIONS[index]: float(probability)
                for index, probability in zip(
                    legal_indices, probabilities.tolist(), strict=True
                )
            }
        )
        result.validate_support(information_set)
        return result

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues:
        _, raw_values = self._outputs(information_set, belief)
        acting_player = int(information_set.info_set_id[1])
        if player not in (0, 1):
            raise ValueError("counterfactual value player must be 0 or 1")
        sign = 1.0 if player == acting_player else -1.0
        by_card = {card.label: float(raw_values[int(card)]) for card in DuelCard}
        return CounterfactualValues(
            player=player,
            values=tuple(
                (
                    hypothesis.hypothesis_id,
                    sign * by_card[hypothesis.hypothesis_id.split(":", 1)[1]],
                )
                for hypothesis in belief.opponent_range.hypotheses
            ),
            source_fingerprint=self.artifact.fingerprint,
        )


@dataclass(frozen=True, slots=True)
class _TrainingBatch:
    information_sets: tuple[InformationSet, ...]
    beliefs: tuple[PublicBeliefState, ...]
    features: Tensor
    legal_mask: Tensor
    policy_targets: Tensor
    regret_targets: Tensor
    value_targets: Tensor
    value_mask: Tensor
    fingerprint: str


def _training_batch(
    game: RocketPrizeDuel,
    teacher_profile: TabularStrategyProfile,
    regret_targets: Mapping[str, Mapping[str, float]],
    *,
    deck_fingerprint: str,
) -> _TrainingBatch:
    encoder = DuelInformationEncoder()
    teacher = ExactTabularBlueprintProvider(
        game,
        teacher_profile,
        deck_fingerprint=deck_fingerprint,
        algorithm="exact_neural_supervision_teacher",
        training_steps=1,
    )
    information_sets = game.information_sets()
    beliefs: list[PublicBeliefState] = []
    feature_rows: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    policies: list[list[float]] = []
    regrets: list[list[float]] = []
    values: list[list[float]] = []
    value_masks: list[list[bool]] = []
    payload = []
    for information_set in information_sets:
        state = game.states_for_information_set(information_set.info_set_id)[0]
        actor = state.acting_player
        assert actor is not None
        belief = game.public_belief(state, player=actor)
        beliefs.append(belief)
        encoded = encoder.encode(information_set, belief)
        mask = encoder.legal_action_mask(information_set)
        policy = teacher_profile.strategy(information_set)
        policy_row = [policy.probability(action) for action in GLOBAL_DUEL_ACTIONS]
        regret_row = [
            float(regret_targets[information_set.info_set_id].get(action, 0.0))
            for action in GLOBAL_DUEL_ACTIONS
        ]
        target_values = dict(
            teacher.counterfactual_values(
                information_set, belief, player=actor
            ).values
        )
        value_row = [0.0] * len(DuelCard)
        value_mask = [False] * len(DuelCard)
        for card in DuelCard:
            identifier = f"opponent:{card.label}"
            if identifier in target_values:
                value_row[int(card)] = target_values[identifier]
                value_mask[int(card)] = True
        feature_rows.append(encoded)
        masks.append(mask)
        policies.append(policy_row)
        regrets.append(regret_row)
        values.append(value_row)
        value_masks.append(value_mask)
        payload.append(
            {
                "information_set": information_set.info_set_id,
                "belief": belief.fingerprint,
                "features": encoded,
                "legal": mask,
                "policy": policy_row,
                "regret": regret_row,
                "values": value_row,
                "value_mask": value_mask,
            }
        )
    return _TrainingBatch(
        information_sets=information_sets,
        beliefs=tuple(beliefs),
        features=torch.tensor(feature_rows, dtype=torch.float64),
        legal_mask=torch.tensor(masks, dtype=torch.bool),
        policy_targets=torch.tensor(policies, dtype=torch.float64),
        regret_targets=torch.tensor(regrets, dtype=torch.float64),
        value_targets=torch.tensor(values, dtype=torch.float64),
        value_mask=torch.tensor(value_masks, dtype=torch.bool),
        fingerprint=_fingerprint(payload),
    )


def _tensor_artifacts(network: DuelBlueprintNetwork) -> tuple[TensorArtifact, ...]:
    result = []
    for name, tensor in network.state_dict().items():
        detached = tensor.detach().cpu().to(torch.float64).contiguous()
        result.append(
            TensorArtifact(
                name=name,
                shape=tuple(detached.shape),
                values=tuple(float(value) for value in detached.reshape(-1).tolist()),
            )
        )
    return tuple(result)


def restore_neural_duel_blueprint(
    game: RocketPrizeDuel,
    artifact: NeuralBlueprintArtifact,
) -> NeuralDuelBlueprintProvider:
    network = DuelBlueprintNetwork(hidden_dim=artifact.hidden_dim)
    expected = network.state_dict()
    supplied = {tensor.name: tensor for tensor in artifact.tensors}
    if set(expected) != set(supplied):
        raise ValueError("neural blueprint tensor names do not match architecture")
    state = {}
    for name, template in expected.items():
        tensor = supplied[name]
        if tuple(template.shape) != tensor.shape:
            raise ValueError(f"tensor shape mismatch for {name}")
        state[name] = torch.tensor(tensor.values, dtype=torch.float64).reshape(tensor.shape)
    network.load_state_dict(state, strict=True)
    return NeuralDuelBlueprintProvider(game, network, artifact)


def provider_profile(
    game: RocketPrizeDuel,
    provider: NeuralDuelBlueprintProvider,
    *,
    label: str,
) -> TabularStrategyProfile:
    strategies = []
    for information_set in game.information_sets():
        state = game.states_for_information_set(information_set.info_set_id)[0]
        actor = state.acting_player
        assert actor is not None
        belief = game.public_belief(state, player=actor)
        strategies.append(
            (information_set.info_set_id, provider.strategy(information_set, belief))
        )
    return TabularStrategyProfile(tuple(strategies), label)


def regret_matching_targets_from_profile(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> dict[str, dict[str, float]]:
    """Encode an average strategy as positive regret-matching supervision.

    This trains the inference parameterization used by a neural CFR-family
    candidate without claiming that supervised distillation itself is Deep CFR.
    """

    return {
        information_set.info_set_id: {
            action.action_id: profile.strategy(information_set).probability(
                action.action_id
            )
            for action in information_set.legal_actions
        }
        for information_set in game.information_sets()
    }


def train_neural_duel_blueprint(
    game: RocketPrizeDuel,
    teacher_profile: TabularStrategyProfile,
    regret_targets: Mapping[str, Mapping[str, float]],
    *,
    deck_fingerprint: str,
    strategy_mode: str,
    seed: int = 20260726,
    epochs: int = 2_000,
    hidden_dim: int = 32,
    learning_rate: float = 0.03,
) -> NeuralBlueprintTrainingRun:
    if strategy_mode not in {"policy", "regret"}:
        raise ValueError("strategy_mode must be policy or regret")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("training epochs/learning rate must be positive")
    _validate_digest(deck_fingerprint, "deck_fingerprint")
    batch = _training_batch(
        game,
        teacher_profile,
        regret_targets,
        deck_fingerprint=deck_fingerprint,
    )
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = DuelBlueprintNetwork(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    final_loss = math.inf
    for _ in range(epochs):
        raw_strategy, raw_values = network(batch.features)
        if strategy_mode == "policy":
            masked_logits = raw_strategy.masked_fill(~batch.legal_mask, -1e9)
            log_probabilities = torch.log_softmax(masked_logits, dim=-1)
            strategy_loss = -(
                batch.policy_targets * log_probabilities
            ).sum(dim=-1).mean()
        else:
            strategy_loss = torch.mean(
                (raw_strategy[batch.legal_mask] - batch.regret_targets[batch.legal_mask])
                ** 2
            )
        value_loss = torch.mean(
            (raw_values[batch.value_mask] - batch.value_targets[batch.value_mask]) ** 2
        )
        loss = strategy_loss + 0.25 * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    artifact = NeuralBlueprintArtifact(
        strategy_mode=strategy_mode,
        hidden_dim=hidden_dim,
        game_fingerprint=game.fingerprint,
        deck_fingerprint=deck_fingerprint,
        encoder_fingerprint=DuelInformationEncoder().fingerprint,
        training_dataset_fingerprint=batch.fingerprint,
        teacher_profile_fingerprint=teacher_profile.fingerprint,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        final_training_loss=final_loss,
        tensors=_tensor_artifacts(network),
    )
    provider = NeuralDuelBlueprintProvider(game, network, artifact)
    profile = provider_profile(
        game, provider, label=f"neural_{strategy_mode}_blueprint"
    )
    metrics = exact_profile_metrics(game, profile)
    policy_l1 = sum(
        abs(
            profile.strategy(info).probability(action.action_id)
            - teacher_profile.strategy(info).probability(action.action_id)
        )
        for info in game.information_sets()
        for action in info.legal_actions
    )
    squared_value_errors = []
    teacher = ExactTabularBlueprintProvider(
        game,
        teacher_profile,
        deck_fingerprint=deck_fingerprint,
        algorithm="exact_value_metric_teacher",
        training_steps=1,
    )
    for info, belief in zip(batch.information_sets, batch.beliefs, strict=True):
        state = game.states_for_information_set(info.info_set_id)[0]
        actor = state.acting_player
        assert actor is not None
        expected = dict(teacher.counterfactual_values(info, belief, player=actor).values)
        predicted = dict(provider.counterfactual_values(info, belief, player=actor).values)
        squared_value_errors.extend(
            (predicted[key] - value) ** 2 for key, value in expected.items()
        )
    value_rmse = math.sqrt(sum(squared_value_errors) / len(squared_value_errors))
    report = NeuralBlueprintTrainingReport(
        strategy_mode=strategy_mode,
        epochs=epochs,
        seed=seed,
        final_training_loss=final_loss,
        policy_l1_to_teacher=policy_l1,
        conditional_value_rmse=value_rmse,
        metrics=metrics,
        artifact_fingerprint=artifact.fingerprint,
        checkpoint_fingerprint=provider.checkpoint.model_fingerprint,
    )
    return NeuralBlueprintTrainingRun(
        provider=provider,
        artifact=artifact,
        profile=profile,
        report=report,
    )
