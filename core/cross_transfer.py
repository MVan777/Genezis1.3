"""
Кросс-Доменный Перенос Знаний (Cross-Domain Knowledge Transfer)
Позволяет переносить обобщенные прототипы нейронной памяти между разными средами
"""

import numpy as np

class CrossDomainTransfer:
    """Модуль переноса концептуальных прототипов между различными задачами"""

    def __init__(self):
        self.transferred_prototypes = []

    def export_prototypes(self, brain):
        """Экспортировать лучшие концептуальные прототипы из мога"""
        prototypes = []
        for cluster in brain.router.clusters:
            for neuron in cluster.neurons:
                if getattr(neuron, 'confidence', 0) > 0.6 or neuron.usage_count > 10:
                    prototypes.append(neuron)
        self.transferred_prototypes = prototypes
        return len(prototypes)

    def import_prototypes(self, brain, target_domain="general"):
        """Импортировать прототипы в соответствующий кластер целевого мозга"""
        imported = 0
        if not self.transferred_prototypes:
            return 0

        target_cluster = brain.router.select_cluster(np.zeros(10))
        for proto in self.transferred_prototypes:
            target_cluster.add_neuron(proto)
            imported += 1
        return imported
