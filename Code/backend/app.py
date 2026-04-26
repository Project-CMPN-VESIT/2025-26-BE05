from flask import Flask
from flask_cors import CORS

from backend.config import PORT
from backend.services.artifacts import load_artifacts
from backend.routes.main import bp as main_bp
from backend.routes.predict import bp as predict_bp
from backend.routes.explain import bp as explain_bp


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(explain_bp)

    return app


app = create_app()

if __name__ == "__main__":
    load_artifacts()
    print("\n" + "=" * 50)
    print("Starting Credit Risk API with Explainability")
    print("=" * 50)
    print(f"Port: {PORT}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=PORT, debug=True)