"""Category parsing and label building utilities for SAE feature analysis."""

import json
from collections import defaultdict
from typing import Any

import numpy as np


def parse_category_field(category_value: Any) -> list[str]:
    """Parse a category field and return category names.
    
    Supported formats:
    - JSON string: '{"harm:XXX": 1.0, "harm:YYY": 1.0}'
    - dict: {"harm:XXX": 1.0}
    - comma-separated string: "harm:XXX, meta:YYY"
    - Empty string or '{}'
    """
    if not category_value or category_value == "{}":
        return []

    if isinstance(category_value, dict):
        return list(category_value.keys())

    if not isinstance(category_value, str):
        return []

    try:
        parsed = json.loads(category_value)
    except (json.JSONDecodeError, TypeError):
        return [c.strip() for c in category_value.split(",") if c.strip()]

    if isinstance(parsed, dict):
        return list(parsed.keys())
    if isinstance(parsed, list):
        return [str(c) for c in parsed if str(c)]
    return []


def build_category_labels(
    data_list: list[dict[str, Any]],
    min_samples: int = 50,
    *,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build one binary label vector per category.
    
    Args:
        data_list: normalized dataset records with a category field.
        min_samples: minimum number of positive samples for a category.
        verbose: print category count summary when True.
        
    Returns:
        A pair of `{category_name: binary_labels}` and full category counts.
    """
    category_counts = defaultdict(int)
    for item in data_list:
        cats = parse_category_field(item.get("category", ""))
        for c in cats:
            category_counts[c] += 1
    
    valid_categories = [c for c, cnt in category_counts.items() if cnt >= min_samples]

    if verbose:
        print(f"Total {len(category_counts)} categories found")
        print(
            f"Among them, {len(valid_categories)} categories satisfy the "
            f"minimum sample number requirement (>= {min_samples})"
        )
    
    category_labels = {}
    for cat in valid_categories:
        y = np.array(
            [1 if cat in parse_category_field(item.get("category", "")) else 0 for item in data_list]
        )
        category_labels[cat] = y
    
    return category_labels, dict(category_counts)
