# settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def _env_list(name, default=None):
    raw = os.getenv(name, "")
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals or (default or [])

# ─────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes", "on")
ALLOWED_HOSTS = _env_list("ALLOWED_HOSTS", ["*"])

APPEND_SLASH = False  # avoid 301/308 on OPTIONS preflight (CORS killer)

# ─────────────────────────────────────────────────────────────
# Apps
# ─────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "corsheaders",  # keep high
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # your apps
    "crawler",
    "chatbot",
    "ingestion",
    "ocr",
    "compliance",

    # third-party
    "rest_framework",
]

# ─────────────────────────────────────────────────────────────
# Middleware (CORS must be before CommonMiddleware)
# ─────────────────────────────────────────────────────────────
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",  # keep if you use cookies/CSRF
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "autonovin.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "autonovin.wsgi.application"

# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME"),
        "USER": os.getenv("DB_USER"),
        "PASSWORD": os.getenv("DB_PASSWORD"),
        "HOST": os.getenv("DB_HOST"),
        "PORT": os.getenv("DB_PORT"),
    }
}

# ─────────────────────────────────────────────────────────────
# Auth / DRF
# ─────────────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        # Keep TokenAuth. If you also want cookie sessions, add SessionAuthentication.
        "rest_framework.authentication.TokenAuthentication",
        # "rest_framework.authentication.SessionAuthentication",
    ],
}

# ─────────────────────────────────────────────────────────────
# I18N / TZ
# ─────────────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ─────────────────────────────────────────────────────────────
# Static
# ─────────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]

# ─────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://redis:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

# ─────────────────────────────────────────────────────────────
# External
# ─────────────────────────────────────────────────────────────
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RABBIT_HOST = os.getenv("RABBIT_HOST")
RABBIT_PORT = os.getenv("RABBIT_PORT")
RABBIT_USER = os.getenv("RABBIT_USER")
RABBIT_PASS = os.getenv("RABBIT_PASS")
COMPLIANCE_QUEUE_SELL = os.getenv("COMPLIANCE_QUEUE_SELL")
COMPLIANCE_QUEUE_BUY = os.getenv("COMPLIANCE_QUEUE_BUY")

# ─────────────────────────────────────────────────────────────
# App specific
# ─────────────────────────────────────────────────────────────
MODEL_DIR = BASE_DIR / "models"
YOLO_WEIGHTS_PATH = MODEL_DIR / "LP-detection.pt"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ─────────────────────────────────────────────────────────────
# CORS (credentials-friendly + local IPs)
# ─────────────────────────────────────────────────────────────
# Exact origins via env (comma-separated), e.g.:
# CORS_ALLOWED_ORIGINS="http://localhost:3000,http://192.168.40.90:3000"
CORS_ALLOWED_ORIGINS = _env_list(
    "CORS_ALLOWED_ORIGINS",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000",
    ],
)

# Also accept common local IP ranges & ports (3000/5173/8080/4200 etc)
CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    r"^https?://(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?$",
    r"^https?://(?:192\.168\.\d{1,3}\.\d{1,3})(?::\d+)?$",
    r"^https?://(?:172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(?::\d+)?$",
    r"^https?://[a-zA-Z0-9-]+\.local(?::\d+)?$",
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_PRIVATE_NETWORK = True  # helps with Chrome PNA on LAN

from corsheaders.defaults import default_headers
CORS_ALLOW_HEADERS = list(default_headers) + [
    "authorization",
    "x-requested-with",
    "x-csrftoken",
    "ngrok-skip-browser-warning",  # harmless to allow if you ever use ngrok
]

# ─────────────────────────────────────────────────────────────
# CSRF / Cookies (for cross-site cookie flows)
# ─────────────────────────────────────────────────────────────
# IMPORTANT: Django requires explicit list (no regex) for CSRF trusted origins.
# Set via env: CSRF_TRUSTED_ORIGINS="http://192.168.40.90:3000,http://localhost:3000"
CSRF_TRUSTED_ORIGINS = _env_list(
    "CSRF_TRUSTED_ORIGINS",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000",
    ],
)

# If you actually use cookies, keep SameSite=None and Secure (HTTPS required).
SESSION_COOKIE_SAMESITE = "None"
CSRF_COOKIE_SAMESITE = "None"
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# In dev, DO NOT force HTTPS redirects (preflight cannot follow scheme changes)
SECURE_SSL_REDIRECT = False
# If you’re behind a proxy that sets X-Forwarded-Proto, set:
# SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
