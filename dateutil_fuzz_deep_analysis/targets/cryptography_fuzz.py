from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, utils

# fuzz_dsa.py 

def fuzz(buf):
    if len(buf) < 40:
        return  # cần ít nhất 40 byte để chia thành 2 phần hợp lệ

    try:
        private_key = dsa.generate_private_key(key_size=1024)
        public_key = private_key.public_key()

        data = buf[:20]
        more_data = buf[20:40]

        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(data)
        hasher.update(more_data)
        digest = hasher.finalize()

        sig1 = private_key.sign(data, hashes.SHA256())
        sig2 = private_key.sign(digest, utils.Prehashed(hashes.SHA256()))

        public_key.verify(sig1, data, hashes.SHA256())
        public_key.verify(sig2, digest, utils.Prehashed(hashes.SHA256()))

    except Exception:
        pass