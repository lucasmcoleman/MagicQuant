"""
Evolutionary Survival - Core evolutionary search algorithm.

This module implements the evolutionary search that:
1. Generates candidate hybrid configurations
2. Classifies them into tiers based on size ratio to baseline
3. Runs tournaments to select winners
4. Applies mutation strategies (Protector/Crusher)
5. Implements epsilon-greedy exploration

The search is designed around MXFP4 as the primary compression scheme,
with the evolutionary pressure finding which tensor groups can tolerate
MXFP4 and which need protection at higher precision.
"""

from typing import Dict, List, Optional
import random
import copy

from magicquant.quant.schemes import (
    get_all_schemes, get_scheme_by_name, get_floor_for_group_class
)


# Ordered from highest quality (lowest noise) to most compressed (highest noise).
# Derived from the canonical scheme registry — see magicquant.quant.schemes.
SCHEME_QUALITY_ORDER: List[str] = [s.name for s in get_all_schemes()]


class EvolutionarySurvivor:
    """
    Run evolutionary search for optimal hybrid quantizations.

    The algorithm follows these phases:
        1. Initialize population with high-probability seeds
        2. Predict performance of all candidates
        3. Classify into tiers by size ratio to baseline
        4. Tournament selection within each tier
        5. Mutation of winners (Protector upgrades brain layers,
           Crusher downgrades robust layers)
        6. Epsilon-greedy exploration
    """

    AVAILABLE_SCHEMES = SCHEME_QUALITY_ORDER

    DEFAULT_GROUPS = ['E', 'H', 'Q', 'K', 'O', 'U', 'D']

    @staticmethod
    def _upgrade(scheme: str) -> Optional[str]:
        """Return the next-better scheme, or None if at the top."""
        try:
            return get_scheme_by_name(scheme).upgrade_neighbor
        except ValueError:
            return None

    @staticmethod
    def _downgrade(scheme: str) -> Optional[str]:
        """Return the next-smaller scheme, or None if at the bottom."""
        try:
            return get_scheme_by_name(scheme).downgrade_neighbor
        except ValueError:
            return None

    # Groups that are sensitive to quantization ("brain" layers)
    _HIGH_SENSITIVITY = {'E', 'H', 'O', 'R'}

    # Groups that are robust to quantization (FFN / experts)
    _LOW_SENSITIVITY = {'U', 'D', 'X'}

    # Floor for each group class — read from registry helper so the
    # values stay consistent if the registry's bottom scheme changes.
    @staticmethod
    def _min_scheme_for_class(group_class: str) -> str:
        """Get the minimum acceptable scheme for a group class
        ("sensitive" or "robust")."""
        return get_floor_for_group_class(group_class)

    def __init__(
        self,
        predictor,
        baseline_config: Dict[str, str],
        max_generations: int = 50,
        population_size: int = 100,
        epsilon: float = 0.2,
        enable_rocmfpx: bool = False,
    ):
        self.predictor = predictor
        self.baseline_config = baseline_config
        self.max_generations = max_generations
        self.population_size = population_size
        self.epsilon = epsilon
        # When True, the AMD-native ROCmFPX fork schemes join the candidate
        # pool (random-config sampling + dedicated seeds). Off by default so
        # the standard search — and its seed-pinned regression fixture — is
        # unchanged. Gated further at encode time by the libggml probe.
        self.enable_rocmfpx = enable_rocmfpx

        self.history: List[Dict] = []
        self.tier_winners: Dict[str, Dict] = {}

    def run_evolution(
        self,
        groups: Optional[List[str]] = None,
        verbose: bool = True,
        patience: Optional[int] = None,
        min_improvement: float = 1e-4,
    ) -> List[Dict]:
        """Run the evolutionary search.

        Args:
            groups: tensor-group keys to vary (defaults to DEFAULT_GROUPS).
            verbose: print per-generation progress.
            patience: if set, stop early when the best composite_score does not
                improve by more than ``min_improvement`` for this many
                consecutive generations. ``None`` (the default) disables
                early-stopping so the full ``max_generations`` budget runs —
                this preserves the historical behavior and keeps the
                seed-pinned refactor-regression fixture stable.
            min_improvement: minimum increase in the best composite_score that
                counts as progress for the patience counter.
        """
        if groups is None:
            groups = self.DEFAULT_GROUPS

        population = self._initialize_population(groups)

        if verbose:
            print(f"Initialized population of {len(population)} candidates")

        best_configs = []
        seen_keys = set()  # O(1) membership instead of O(n*m) re-serialization

        best_score_so_far = float("-inf")
        gens_without_improvement = 0

        for generation in range(self.max_generations):
            predictions = self._predict_population(population)
            tier_assignment = self._classify_into_tiers(predictions)
            winners = self._tournament_selection(tier_assignment)

            if verbose and (generation + 1) % 5 == 0:
                print(f"Gen {generation+1}: {len(winners)} tier winners, "
                      f"{len(best_configs)} unique configs discovered")

            for winner in winners:
                config_key = str(sorted(winner['config'].items()))
                if config_key not in seen_keys:
                    seen_keys.add(config_key)
                    best_configs.append(winner)

            # Early-stopping: track the best composite_score this generation and
            # stop if it plateaus for `patience` generations. Disabled by
            # default (patience is None) to keep the full-budget behavior.
            if patience is not None:
                gen_best = max(
                    (w.get('composite_score', float("-inf")) for w in winners),
                    default=float("-inf"),
                )
                if gen_best > best_score_so_far + min_improvement:
                    best_score_so_far = gen_best
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1
                    if gens_without_improvement >= patience:
                        if verbose:
                            print(f"Early stop at gen {generation+1}: no "
                                  f"improvement for {patience} generations")
                        break

            # Mutation: Protector upgrades brain layers, Crusher downgrades FFN
            mutants = self._mutate_winners(winners, groups)
            # Also carry forward the winners themselves
            population = [{'config': copy.deepcopy(w['config'])} for w in winners]
            population.extend(mutants)

            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                population.extend(self._generate_exploration_configs(groups, n=10))

            # Fill remaining slots with random configs
            while len(population) < self.population_size:
                population.append({'config': self._generate_random_config(groups)})

            population = population[:self.population_size]

        best_configs.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        return best_configs

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, groups: List[str]) -> List[Dict]:
        population = []

        # Seed 1: MagicQuant signature — brain protected, FFN at MXFP4
        mxfp4_core = {g: "MXFP4_MOE" for g in groups}
        mxfp4_core['E'] = "BF16"
        mxfp4_core['H'] = "BF16"
        mxfp4_core['O'] = "Q8_0"
        population.append({'config': mxfp4_core})

        # Seed 2: Maximum MXFP4 — everything except embeddings
        mxfp4_max = {g: "MXFP4_MOE" for g in groups}
        mxfp4_max['E'] = "BF16"
        mxfp4_max['H'] = "BF16"
        population.append({'config': mxfp4_max})

        # Seed 3: MXFP4 FFN with IQ4_NL attention
        mxfp4_iq4 = {g: "IQ4_NL" for g in groups}
        mxfp4_iq4['E'] = "BF16"
        mxfp4_iq4['H'] = "BF16"
        mxfp4_iq4['U'] = "MXFP4_MOE"
        mxfp4_iq4['D'] = "MXFP4_MOE"
        population.append({'config': mxfp4_iq4})

        # Seed 4: High-Contrast — brain at BF16, attention at Q6_K, FFN at MXFP4
        high_contrast = {g: "Q6_K" for g in groups}
        high_contrast['E'] = "BF16"
        high_contrast['H'] = "BF16"
        high_contrast['O'] = "Q8_0"
        high_contrast['U'] = "MXFP4_MOE"
        high_contrast['D'] = "MXFP4_MOE"
        population.append({'config': high_contrast})

        # Seed 5: Uniform schemes for reference
        for scheme in ["Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE"]:
            population.append({'config': {g: scheme for g in groups}})

        # Seed 5b (opt-in): ROCmFPX-native seeds so the AMD family starts near
        # good configs rather than only arriving via random sampling.
        if self.enable_rocmfpx:
            # Brain-protected fp4 core (the MagicQuant signature, AMD-native).
            rocm_core = {g: "ROCMFP4" for g in groups}
            rocm_core['E'] = "BF16"
            rocm_core['H'] = "BF16"
            rocm_core['O'] = "ROCMFP8"
            population.append({'config': rocm_core})
            # High-contrast: fp6 attention, fp4 FFN, protected brain.
            rocm_contrast = {g: "ROCMFP6" for g in groups}
            rocm_contrast['E'] = "BF16"
            rocm_contrast['H'] = "BF16"
            rocm_contrast['U'] = "ROCMFP4"
            rocm_contrast['D'] = "ROCMFP4"
            population.append({'config': rocm_contrast})
            # Uniform references across the family.
            for scheme in ["ROCMFP8", "ROCMFP6", "ROCMFP4"]:
                population.append({'config': {g: scheme for g in groups}})

        # Seed 6: Random configs weighted toward MXFP4
        for _ in range(self.population_size - len(population)):
            population.append({'config': self._generate_random_config(groups)})

        return population[:self.population_size]

    # Sampling weights per group class, indexed by scheme category.
    # Each value is the relative probability mass for picking ANY scheme
    # in that category. Within a category, we further weight inversely by
    # noise_factor so higher-quality variants are preferred slightly over
    # lower-quality ones in the same category.
    #
    # _BRAIN_CLASS_WEIGHTS: for high-sensitivity groups (E, H, O, R) —
    #   biased toward float and high-precision schemes.
    # _ATTENTION_CLASS_WEIGHTS: for moderate-sensitivity groups (Q, K) —
    #   middle-ground spread.
    # _FFN_CLASS_WEIGHTS: for robust groups (U, D, X) —
    #   biased toward maximum compression.
    _BRAIN_CLASS_WEIGHTS = {
        "float":    0.30,   # BF16
        "legacy_q": 0.30,   # Q8_0
        "k_quant":  0.30,   # Q6_K, Q5_K, Q4_K_M, Q3_K, Q2_K
        "iq_quant": 0.05,   # IQ4_NL (and any IQ-quants added later)
        "mxfp4":    0.05,   # MXFP4_MOE
    }
    _ATTENTION_CLASS_WEIGHTS = {
        "float":    0.05,
        "legacy_q": 0.15,
        "k_quant":  0.45,
        "iq_quant": 0.20,
        "mxfp4":    0.15,
    }
    _FFN_CLASS_WEIGHTS = {
        "float":    0.02,
        "legacy_q": 0.05,
        "k_quant":  0.30,
        "iq_quant": 0.30,
        "mxfp4":    0.33,
    }

    # ROCmFPX mass added to each class dict when enable_rocmfpx is set. The
    # AMD-native family is offered as a compression alternative — a little for
    # brain groups (fp8/fp6 as high-quality options), more for FFN (fp4 is the
    # fork's fastest path). Merged, not replacing, so the search compares
    # rocmfpx head-to-head against the standard schemes via the predictor.
    _ROCMFPX_CLASS_MASS = {
        "brain":     0.10,
        "attention": 0.25,
        "ffn":       0.40,
    }

    def _class_weights(self, group_key: str) -> Dict[str, float]:
        """Return the category-weight dict for a group class ('brain',
        'attention', 'ffn'), injecting rocmfpx mass when enabled."""
        base = {
            "brain": self._BRAIN_CLASS_WEIGHTS,
            "attention": self._ATTENTION_CLASS_WEIGHTS,
            "ffn": self._FFN_CLASS_WEIGHTS,
        }[group_key]
        if not self.enable_rocmfpx:
            return base
        merged = dict(base)
        merged["rocmfpx"] = self._ROCMFPX_CLASS_MASS[group_key]
        return merged

    def _generate_random_config(self, groups: List[str]) -> Dict[str, str]:
        """Generate a random config biased toward compression for FFN and
        higher precision for brain layers.

        Weights are category-indexed (not positional) so adding new schemes
        to the registry doesn't require updating positional arrays.
        """
        from magicquant.quant.schemes import get_all_schemes, ROCMFPX_SCHEME_NAMES

        config: Dict[str, str] = {}
        all_schemes = get_all_schemes()
        if not self.enable_rocmfpx:
            # Defense-in-depth: rocmfpx categories already carry zero weight
            # when disabled, but drop the schemes entirely so a future weight
            # typo can't leak an unusable fork type into a standard search.
            all_schemes = [s for s in all_schemes if s.name not in ROCMFPX_SCHEME_NAMES]

        for g in groups:
            if g in self._HIGH_SENSITIVITY:
                class_weights = self._class_weights("brain")
            elif g in self._LOW_SENSITIVITY:
                class_weights = self._class_weights("ffn")
            else:
                class_weights = self._class_weights("attention")

            # Build per-scheme weights: start with the class weight, then
            # divide it across all schemes in that category, inversely
            # weighted by noise_factor (cleaner schemes preferred).
            scheme_weights = []
            for s in all_schemes:
                cat_weight = class_weights.get(s.category, 0.0)
                # Within a category, give cleaner (lower noise) schemes more weight.
                # noise_factor=0 (BF16) gets factor 2.0; high-noise gets factor near 0.
                # Normalize within a category later.
                scheme_weights.append(cat_weight * (1.0 / (1.0 + s.noise_factor)))

            # Avoid all-zeros pathology
            if sum(scheme_weights) == 0:
                scheme_weights = [1.0] * len(all_schemes)

            picked = random.choices(
                [s.name for s in all_schemes],
                weights=scheme_weights,
            )[0]

            # Enforce the sensitive-group floor (Q8_0): high-sensitivity
            # "brain" groups (E, H, O, R) must never be sampled below the
            # floor. The class weights still allow low-bit picks (mxfp4/k_quant
            # in _BRAIN_CLASS_WEIGHTS), so clamp here rather than silently
            # producing a sub-floor sensitive config the Protector can't fix.
            if g in self._HIGH_SENSITIVITY:
                floor = self._min_scheme_for_class('sensitive')
                try:
                    picked_bpw = get_scheme_by_name(picked).bits_per_weight
                    floor_bpw = get_scheme_by_name(floor).bits_per_weight
                    if picked_bpw < floor_bpw:
                        picked = floor
                except ValueError:
                    pass

            config[g] = picked
        return config

    # ------------------------------------------------------------------
    # Prediction and tier classification
    # ------------------------------------------------------------------

    def _predict_population(self, population: List[Dict]) -> List[Dict]:
        for candidate in population:
            scores = self.predictor.score_hybrid(candidate['config'])
            candidate.update(scores)
        return population

    def _classify_into_tiers(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify into tiers using the canonical tier boundaries from
        ``magicquant.quant.tiers.classify_tier`` to ensure consistency
        between evolutionary search and final survivor selection."""
        from magicquant.quant.tiers import classify_tier

        baseline_gb = self.predictor.baseline_size_gb
        tier_assignment: Dict[str, List[Dict]] = {}

        for pred in predictions:
            size_gb = pred.get('predicted_size_gb', 1.0)
            tier = classify_tier(size_gb, baseline_gb)

            if tier not in tier_assignment:
                tier_assignment[tier] = []
            tier_assignment[tier].append(pred)

        return tier_assignment

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------

    def _tournament_selection(self, tier_assignment: Dict) -> List[Dict]:
        winners = []
        for tier, candidates in tier_assignment.items():
            if not candidates:
                continue
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('composite_score', 0),
                reverse=True
            )
            tier_winners = sorted_candidates[:min(3, len(sorted_candidates))]
            for w in tier_winners:
                w['tier'] = tier
                self.tier_winners[tier] = w
            winners.extend(tier_winners)
        return winners

    # ------------------------------------------------------------------
    # Mutation (Protector / Crusher)
    # ------------------------------------------------------------------

    def _mutate_winners(self, winners: List[Dict], groups: List[str]) -> List[Dict]:
        population = []
        for winner in winners:
            config = copy.deepcopy(winner['config'])

            # Protector: upgrade the most sensitive unprotected brain layer
            target = self._find_protector_target(config, groups)
            if target:
                new_scheme = self._upgrade(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})

            # Crusher: downgrade the most robust high-precision FFN layer
            target = self._find_crusher_target(config, groups)
            if target:
                new_scheme = self._downgrade(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})

            # Swap mutant: try MXFP4 on the attention layers
            for g in ['Q', 'K']:
                if g in config and config[g] != "MXFP4_MOE":
                    c = config.copy()
                    c[g] = "MXFP4_MOE"
                    population.append({'config': c})

        return population

    def _find_protector_target(
        self, config: Dict[str, str], groups: List[str]
    ) -> Optional[Dict]:
        """Find the most sensitive brain layer that isn't at max precision."""
        candidates = []
        for g in groups:
            if g not in self._HIGH_SENSITIVITY:
                continue
            scheme = config.get(g, "Q8_0")
            if scheme != "BF16":
                sensitivity = self.predictor.sensitivity_weights.get(g, 0.5)
                candidates.append({
                    'group': g, 'scheme': scheme, 'sensitivity': sensitivity
                })
        if not candidates:
            return None
        candidates.sort(key=lambda x: x['sensitivity'], reverse=True)
        return candidates[0]

    def _find_crusher_target(
        self, config: Dict[str, str], groups: List[str]
    ) -> Optional[Dict]:
        """Find the most robust FFN layer that's above minimum precision."""
        candidates = []
        for g in groups:
            if g not in self._LOW_SENSITIVITY:
                continue
            scheme = config.get(g, "MXFP4_MOE")
            # Can we push it lower? (Skip if already at the bottom of registry)
            if self._downgrade(scheme) is not None and scheme != self._min_scheme_for_class('robust'):
                sensitivity = self.predictor.sensitivity_weights.get(g, 0.5)
                candidates.append({
                    'group': g, 'scheme': scheme, 'sensitivity': sensitivity
                })
        if not candidates:
            return None
        candidates.sort(key=lambda x: x['sensitivity'])
        return candidates[0]

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _generate_exploration_configs(
        self, groups: List[str], n: int = 10
    ) -> List[Dict]:
        return [{'config': self._generate_random_config(groups)} for _ in range(n)]

    def get_best_config_per_tier(self) -> Dict[str, Dict[str, str]]:
        return self.tier_winners.copy()

    def get_discovered_configs(self, limit: int = 20) -> List[Dict]:
        return self.history[:limit]
