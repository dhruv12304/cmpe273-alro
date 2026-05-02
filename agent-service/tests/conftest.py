import os

os.environ.setdefault("GRID_SIZE", "5")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("QTABLE_PATH", "/tmp/test_qtable.npy")
os.environ.setdefault("INVENTORY_SERVICE_URL", "http://localhost:8002")
os.environ.setdefault("TRAINING_EPISODES", "500")
os.environ.setdefault("CACHE_TTL_SECONDS", "60")
