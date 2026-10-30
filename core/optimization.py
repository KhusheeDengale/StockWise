"""Genetic Algorithm (GA) optimizer for explainable multi-product inventory replenishment.
"""
import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import GA_CONFIG, OBJECTIVE_WEIGHTS
from core.greedy_baseline import (
    compute_normalization_scales,
    compute_objective_components,
    repair_chromosome,
    simulate_inventory_for_product,
    solve_greedy_baseline,
)
from core.schemas import OptimizationMetrics, OptimizationResult


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm purchase optimizer solving multi-product constrained inventory ordering."""

    def __init__(
        self,
        population_size: int = GA_CONFIG["population_size"],
        generations: int = GA_CONFIG["generations"],
        tournament_size: int = GA_CONFIG["tournament_size"],
        crossover_prob: float = GA_CONFIG["crossover_probability"],
        mutation_prob: float = GA_CONFIG["mutation_probability"],
        elite_count: int = GA_CONFIG["elite_count"],
        random_seed: int = GA_CONFIG["random_seed"],
        patience: int = GA_CONFIG["early_stopping_patience"],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.pop_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.random_seed = random_seed
        self.patience = patience
        self.weights = weights if weights is not None else OBJECTIVE_WEIGHTS

    def optimize(
        self,
        products_data: List[Dict[str, Any]],
        budget_gbp: float,
        greedy_baseline_result: Optional[OptimizationResult] = None,
    ) -> OptimizationResult:
        """Run the complete Genetic Algorithm purchase optimization.
        
        Args:
            products_data: List of product specifications.
            budget_gbp: Total budget in GBP.
            greedy_baseline_result: Optional precomputed greedy result.
            
        Returns:
            OptimizationResult containing best chromosome, metrics, and event log.
        """
        start_time = time.time()
        rng = np.random.RandomState(self.random_seed)
        n_products = len(products_data)

        # 1. Compute normalization scales
        scales = compute_normalization_scales(products_data, budget_gbp)

        # 2. Get greedy baseline if not provided
        if greedy_baseline_result is None:
            greedy_baseline_result = solve_greedy_baseline(
                products_data, budget_gbp, scales, self.weights
            )

        greedy_chrom = greedy_baseline_result.chromosome

        # 3. Initialize Population
        population: List[List[int]] = []
        
        # Individual 0: Zero-order chromosome
        zero_chrom, _ = repair_chromosome([0] * n_products, products_data, budget_gbp)
        population.append(zero_chrom)

        # Individual 1: Feasible greedy chromosome
        rep_greedy, _ = repair_chromosome(greedy_chrom, products_data, budget_gbp)
        population.append(rep_greedy)

        # Remaining individuals: Random feasible initialization
        sample_repair_event: Optional[Dict[str, Any]] = None

        for _ in range(self.pop_size - 2):
            raw_ind = []
            for p in products_data:
                multiple = max(1, p["OrderMultiple"])
                curr_stock = p["CurrentStock"]
                storage_cap = p["StorageCapacity"]
                cost = p["SupplierCostGBP"]
                
                max_storage_q = max(0, storage_cap - curr_stock)
                max_budget_q = int(budget_gbp // max(cost, 0.1))
                upper_bound = min(max_storage_q, max_budget_q)
                
                if upper_bound > 0:
                    max_steps = upper_bound // multiple
                    steps = rng.randint(0, max(max_steps + 1, 2))
                    raw_ind.append(steps * multiple)
                else:
                    raw_ind.append(0)

            repaired_ind, repair_actions = repair_chromosome(raw_ind, products_data, budget_gbp)
            if sample_repair_event is None and repair_actions:
                sample_repair_event = {
                    "original_chromosome": raw_ind[:5],
                    "repaired_chromosome": repaired_ind[:5],
                    "actions": repair_actions,
                }
            population.append(repaired_ind)

        initial_population_summary = {
            "size": len(population),
            "includes_zero_order": True,
            "includes_greedy": True,
            "random_individuals": len(population) - 2,
        }

        # 4. Evaluate Population
        scores = [
            compute_objective_components(ind, products_data, budget_gbp, scales, self.weights)[0]
            for ind in population
        ]

        # Tracking events
        sample_selection_event: Optional[Dict[str, Any]] = None
        sample_crossover_event: Optional[Dict[str, Any]] = None
        sample_mutation_event: Optional[Dict[str, Any]] = None

        generation_history = []
        best_overall_obj = float("inf")
        best_overall_chrom = list(population[0])
        generations_without_improvement = 0

        # 5. Evolution Loop
        for gen in range(self.generations):
            # Sort population by score (ascending: minimization)
            sorted_indices = np.argsort(scores)
            population = [population[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            current_best_obj = scores[0]
            current_best_chrom = population[0]
            mean_obj = float(np.mean(scores))

            # Record generation stats
            generation_history.append({
                "generation": gen + 1,
                "best_objective": round(current_best_obj, 6),
                "mean_objective": round(mean_obj, 6),
                "best_chromosome": list(current_best_chrom),
                "feasible_population_count": len(population),
            })

            # Check improvement for early stopping
            if current_best_obj < best_overall_obj - 1e-6:
                best_overall_obj = current_best_obj
                best_overall_chrom = list(current_best_chrom)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.patience:
                break

            # Elitism: preserve top 2
            next_generation = [list(population[0]), list(population[1])]

            # Breed remaining individuals
            while len(next_generation) < self.pop_size:
                # Tournament Selection
                p1_idx = self._tournament_select(scores, rng)
                p2_idx = self._tournament_select(scores, rng)
                parent1 = population[p1_idx]
                parent2 = population[p2_idx]

                if sample_selection_event is None:
                    sample_selection_event = {
                        "tournament_size": self.tournament_size,
                        "selected_parent_1_obj": round(scores[p1_idx], 6),
                        "selected_parent_2_obj": round(scores[p2_idx], 6),
                    }

                # Crossover
                if n_products >= 2 and rng.rand() < self.crossover_prob:
                    cx_point = rng.randint(1, n_products)
                    child1 = parent1[:cx_point] + parent2[cx_point:]
                    child2 = parent2[:cx_point] + parent1[cx_point:]

                    if sample_crossover_event is None:
                        sample_crossover_event = {
                            "crossover_point": cx_point,
                            "parent1_prefix": parent1[:cx_point],
                            "parent2_suffix": parent2[cx_point:],
                            "child1": child1,
                        }
                else:
                    child1 = list(parent1)
                    child2 = list(parent2)

                # Mutation
                for child in (child1, child2):
                    for gene_idx in range(n_products):
                        if rng.rand() < self.mutation_prob:
                            multiple = max(1, products_data[gene_idx]["OrderMultiple"])
                            r_type = rng.rand()
                            old_val = child[gene_idx]

                            if r_type < 0.30:
                                child[gene_idx] = 0
                            elif r_type < 0.65:
                                child[gene_idx] += multiple
                            else:
                                child[gene_idx] = max(0, child[gene_idx] - multiple)

                            if sample_mutation_event is None:
                                sample_mutation_event = {
                                    "product_index": gene_idx,
                                    "original_quantity": old_val,
                                    "mutated_quantity": child[gene_idx],
                                    "order_multiple": multiple,
                                }

                    # Always repair after crossover and mutation
                    rep_child, _ = repair_chromosome(child, products_data, budget_gbp)
                    if len(next_generation) < self.pop_size:
                        next_generation.append(rep_child)

            population = next_generation
            scores = [
                compute_objective_components(ind, products_data, budget_gbp, scales, self.weights)[0]
                for ind in population
            ]

        # Final Evaluation and Best Chromosome
        sorted_indices = np.argsort(scores)
        best_chrom = population[sorted_indices[0]]
        best_obj, final_components = compute_objective_components(
            best_chrom, products_data, budget_gbp, scales, self.weights
        )

        # Invariant Guarantee: GA objective <= greedy objective
        greedy_obj = greedy_baseline_result.metrics.ObjectiveValue
        if best_obj > greedy_obj + 1e-6:
            # Fallback to greedy if numerical discrepancy occurred
            best_chrom = list(greedy_chrom)
            best_obj, final_components = compute_objective_components(
                best_chrom, products_data, budget_gbp, scales, self.weights
            )

        runtime = time.time() - start_time
        total_cost = sum(q * p["SupplierCostGBP"] for q, p in zip(best_chrom, products_data))
        budget_rem = max(0.0, budget_gbp - total_cost)
        util_pct = (total_cost / max(budget_gbp, 1.0)) * 100.0

        total_short = 0.0
        total_exc = 0.0
        for q, p in zip(best_chrom, products_data):
            short, end_stk, _, m_fc = simulate_inventory_for_product(
                q=q,
                current_stock=p["CurrentStock"],
                supplier_cost=p["SupplierCostGBP"],
                lead_time_days=p["LeadTimeDays"],
                daily_forecast=p["DailyForecast"],
            )
            total_short += short
            total_exc += max(0.0, end_stk - (3.0 * m_fc))

        metrics = OptimizationMetrics(
            ObjectiveValue=round(best_obj, 6),
            TotalPurchaseCost=round(total_cost, 2),
            BudgetRemaining=round(budget_rem, 2),
            TotalExpectedShortage=round(total_short, 2),
            TotalExpectedExcess=round(total_exc, 2),
            CoverageImbalance=round(final_components["CoverageImbalance"], 4),
            BudgetUtilizationPercent=round(util_pct, 2),
            RuntimeSeconds=round(runtime, 4),
            Feasible=True,
        )

        prod_orders = {
            p["StockCode"]: q for q, p in zip(best_chrom, products_data)
        }

        explanation_events = {
            "initial_population_summary": initial_population_summary,
            "sample_selection_event": sample_selection_event or {"status": "Deterministic tournament winner preserved"},
            "sample_crossover_event": sample_crossover_event or {"status": "Single product or no crossover triggered"},
            "sample_mutation_event": sample_mutation_event or {"status": "No mutation occurred on sampled step"},
            "sample_repair_event": sample_repair_event or {"status": "Initial population generated within constraints"},
            "final_objective_components": final_components,
        }

        return OptimizationResult(
            algorithm="Genetic Algorithm",
            metrics=metrics,
            chromosome=best_chrom,
            product_orders=prod_orders,
            objective_components=final_components,
            generation_history=generation_history,
            explanation_events=explanation_events,
        )

    def _tournament_select(self, scores: List[float], rng: np.random.RandomState) -> int:
        """Tournament selection: choose best individual out of tournament_size candidates."""
        candidates = rng.choice(len(scores), size=self.tournament_size, replace=False)
        best_cand = candidates[0]
        for c in candidates[1:]:
            if scores[c] < scores[best_cand]:
                best_cand = c
        return int(best_cand)
