"""Code-only remote environment pool service.

Keep this package initializer side-effect free so ``python -m
env.server.pool_server`` executes exactly one module instance on the head
node. Import concrete pool types from ``env.server.pool_server`` instead.
"""
