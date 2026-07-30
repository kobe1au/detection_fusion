"""API, graph, and Manifest malware-detection package.

The package initializer deliberately performs no eager imports. CARE-Droid
and registered comparison methods have separate runtime dependency graphs;
callers import the concrete model, dataset, or loss module they use.
"""

__all__: list[str] = []
