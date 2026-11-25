import os
from flask import Flask
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me")
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
    from .routes import bp
    app.register_blueprint(bp)

    @app.context_processor
    def inject_config():
        return {"config_model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")}

    @app.errorhandler(413)
    def too_large(_error):
        return {"error": "Upload exceeds configured maximum size."}, 413
    return app
