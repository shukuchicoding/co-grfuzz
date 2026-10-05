import dateutil.tz

# fuzz_tzstr.py

def fuzz(buf):
    try:
        # Giải mã đầu vào thành chuỗi Unicode
        data = buf.decode('utf-8', errors='ignore')
        dateutil.tz.tzstr(data)
    except Exception:
        pass