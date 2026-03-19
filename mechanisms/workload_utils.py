def subsample_candidates(candidates, requested_count, prng):
    """Subsample candidates without failing when the request exceeds availability."""
    if requested_count is None:
        return list(candidates)

    selected_count = min(len(candidates), requested_count)
    if selected_count == len(candidates):
        return list(candidates)

    indices = prng.choice(len(candidates), selected_count, replace=False)
    return [candidates[i] for i in indices]
