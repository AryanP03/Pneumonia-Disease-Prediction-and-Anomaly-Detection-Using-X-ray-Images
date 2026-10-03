import importlib
import os

# Dynamically import 1_app without Python identifier syntax limitations
backend_module = importlib.import_module("app.1_app")
app = backend_module.app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)
