"""
Двухпроцессная Модель Мышления Канемана (System 1 & System 2 Cognition Engine)
System 1: Быстрое чутье (<1 мс) для знакомых рутинных ситуаций
System 2: Глубокий 10-шаговый волновой анализ во мнении для сложных моментов
"""

import time
import numpy as np

class SystemDualProcessEngine:
    """Двухпроцессный роутер мышления ИИ (System 1 vs System 2)"""

    def __init__(self, confidence_threshold=0.25, deep_reflection_depth=10):
        self.confidence_threshold = confidence_threshold
        self.deep_reflection_depth = deep_reflection_depth
        self.stats = {'system1_fast_hits': 0, 'system2_deep_reflections': 0}

    def process_decision(self, brain_ref, similar_neurons, confidence):
        """
        Принимает решение по ветви System 1 или System 2
        """
        # Если уверенность выше порога - включается Быстрая System 1 (<1 мс)
        if confidence >= self.confidence_threshold:
            self.stats['system1_fast_hits'] += 1
            return 'system1', confidence, None

        # Иначе включается System 2: Глубокое размышление с 10-шаговым волновым моделированием
        self.stats['system2_deep_reflections'] += 1

        deep_scores = {}
        for neuron, sim, weight in similar_neurons:
            action = neuron.action
            deep_score = self._deep_wave_propagation(neuron, brain_ref, depth=1, max_depth=self.deep_reflection_depth)
            if action not in deep_scores:
                deep_scores[action] = 0.0
            deep_scores[action] += deep_score * sim * weight

        if deep_scores:
            best_action = max(deep_scores.items(), key=lambda x: x[1])[0]
            boosted_confidence = confidence + 0.3
            return 'system2', boosted_confidence, best_action

        return 'system1', confidence, None

    def _deep_wave_propagation(self, start_neuron, brain_ref, depth=1, max_depth=10):
        """Рекурсивный волновой разбор графа памяти до max_depth"""
        if depth >= max_depth or not hasattr(start_neuron, 'next_associations') or not start_neuron.next_associations:
            return start_neuron.flag * start_neuron.strength

        score = start_neuron.flag * start_neuron.strength
        decay = 0.8 ** depth

        for next_id, edge_w in start_neuron.next_associations.items():
            next_n = brain_ref._find_neuron_by_id(next_id)
            if next_n:
                score += decay * edge_w * self._deep_wave_propagation(next_n, brain_ref, depth + 1, max_depth)

        return score
