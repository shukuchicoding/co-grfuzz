import yaml

def fuzz(buf):
    try:
        yaml.safe_load(buf)
    except Exception:
        pass