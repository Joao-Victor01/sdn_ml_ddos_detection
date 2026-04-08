"""
Helpers de plot para uso em ambiente headless.

O pipeline não deve depender de pyplot durante o import dos módulos,
porque isso pode atrasar bastante a inicialização em alguns ambientes.
"""

from __future__ import annotations


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt
