import yaml
import idna
def fuzz(buf):
    try:
        yaml.safe_load(buf)
    except yaml.YAMLError:
        pass

fuzz("0b: X")

yaml.safe_load("0b__:")
# idna.decode(b'\x8d\xd2')