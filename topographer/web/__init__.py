"""Flask + Jinja viewer. Thin on purpose: this package may import ``pipeline``
and ``store``, never ``sources`` or rasterio-adjacent code directly. See
``pipeline.sample.default_sampler`` for where concrete source adapters are
assembled instead.
"""
