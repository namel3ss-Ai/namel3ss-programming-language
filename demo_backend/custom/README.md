# Custom Backend Extensions

This folder is reserved for your handcrafted FastAPI routes and helpers. The
code generator creates it once and will not overwrite files you add here.

- Put reusable dependencies in `__init__.py` or new modules.
- Add route overrides in `routes/custom_api.py` and register them on the
  module-level `router` instance.
- Use the optional `setup(app)` hook to run initialization logic after the
  generated routers are attached (for example, authentication, middleware, or
  event handlers).

Whenever you run the Namel3ss generator again your custom code stays intact.
Refer to the generated modules under `generated/` for available helpers.
