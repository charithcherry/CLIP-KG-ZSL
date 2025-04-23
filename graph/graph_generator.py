import os
import json
import itertools


class GraphGenerator:
    def __init__(self, classes_path, predicates_path, annotation_continuous_path):
        self.classes = self._load_classes(classes_path)
        self.predicates = self._load_predicates(predicates_path)
        self.annotation_continuous_list = self._load_annotation_matrix(annotation_continuous_path)
        self.class_to_annotation_continuous = dict(zip(self.classes, self.annotation_continuous_list))

        self.selected_classes = []
        self.selected_predicates = []
        self.selected_indices = []

    def _load_classes(self, path):
        with open(path, "r") as f:
            return [line.split("\t")[1].strip() for line in f]

    def _load_predicates(self, path):
        with open(path, "r") as f:
            return [line.split("\t")[1].strip() for line in f]

    def _load_annotation_matrix(self, path):
        matrix = []
        with open(path, "r") as f:
            for line in f:
                values = [float(x) for x in line.strip().split(" ") if x != ""]
                matrix.append(values)
        return matrix

    def set_selection(self, selected_classes, selected_predicates):
        self.selected_classes = selected_classes
        self.selected_predicates = selected_predicates
        self.selected_indices = [self.predicates.index(p) for p in selected_predicates]

    def generate_edges(self, MAX_DIFF=10, MIN_DIFF=45, ALPHA=1.20, BETA=0.75, weight_threshold=0.50):
        edges = []

        for u, v in itertools.combinations(self.selected_classes, 2):
            vec_u = self.class_to_annotation_continuous[u]
            vec_v = self.class_to_annotation_continuous[v]

            comparison = []
            contrasting_attrs = []
            num_similar = 0
            num_contrasts = 0

            for i in self.selected_indices:
                val_u = vec_u[i]
                val_v = vec_v[i]
                diff = abs(val_u - val_v)

                if diff <= MAX_DIFF:
                    comparison.append({
                        "predicate": self.predicates[i],
                        "value_cls1": round(val_u, 2),
                        "value_cls2": round(val_v, 2)
                    })
                    num_similar += 1
                elif diff >= MIN_DIFF:
                    contrasting_attrs.append({
                        "predicate": self.predicates[i],
                        "value_cls1": round(val_u, 2),
                        "value_cls2": round(val_v, 2),
                        "difference": round(diff, 2)
                    })
                    comparison.append({
                        "predicate": self.predicates[i],
                        "value_cls1": 0.0,
                        "value_cls2": 0.0
                    })
                    num_contrasts += 1
                else:
                    comparison.append({
                        "predicate": self.predicates[i],
                        "value_cls1": 0.0,
                        "value_cls2": 0.0
                    })

            total_attrs = len(self.selected_indices)
            raw_weight = (ALPHA * num_similar - BETA * num_contrasts) / total_attrs
            weight = round(max(0, min(raw_weight, 1)), 4)

            if num_similar > 0 and weight > weight_threshold:
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": weight,
                    "num_similar": num_similar,
                    "num_contrasts": num_contrasts,
                    "relationship": comparison,
                    "contrasts": contrasting_attrs
                })

        return edges

    def filter_by_predicate_threshold(self, edges, threshold=60):
        filtered = []
        for edge in edges:
            new_relationship = [r for r in edge["relationship"] if r["value_cls1"] >= threshold or r["value_cls2"] >= threshold]
            new_contrasts = [c for c in edge["contrasts"] if c["value_cls1"] >= threshold or c["value_cls2"] >= threshold]
            if new_relationship or new_contrasts:
                edge_copy = edge.copy()
                edge_copy["relationship"] = new_relationship
                edge_copy["contrasts"] = new_contrasts
                filtered.append(edge_copy)
        return filtered

    def match(self, edges, pair_list):
        pair_set = set(frozenset(pair) for pair in pair_list)
        return [edge for edge in edges if frozenset([edge["source"], edge["target"]]) in pair_set]

    def save_edges(self, edges, output_path):
        with open(output_path, "w") as f:
            json.dump(edges, f, indent=4)
        print(f"Saved {len(edges)} edges to: {output_path}")
