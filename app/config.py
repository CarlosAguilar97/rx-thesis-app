import os
from dotenv import load_dotenv

# Solo carga .env si existe (desarrollo local)
if os.path.exists(".env"):
    load_dotenv()

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs_infer")

if not TURSO_DATABASE_URL:
    raise ValueError("TURSO_DATABASE_URL no está configurado")

if not TURSO_AUTH_TOKEN:
    raise ValueError("TURSO_AUTH_TOKEN no está configurado")