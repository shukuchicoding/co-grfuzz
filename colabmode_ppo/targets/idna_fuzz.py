import idna

# fuzz_encode_decode.py 

def fuzz(buf):
    try:
        s = buf.decode('utf-8', errors='ignore')

        # Test encode
        try:
            idna.encode(s)
        except idna.IDNAError:
            pass

        # Test decode
        try:
            idna.decode(s)
        except idna.IDNAError:
            pass

    except Exception:
        pass