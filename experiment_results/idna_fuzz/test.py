import idna
SIZE = 16384
header = [0x32, 0xdf, 0x8d]
middle = [0xe2, 0x80, 0x8c] * SIZE
footer = [0xdf, 0x8c]
b = bytes(header) + bytes(middle) + bytes(footer)
d = b.decode('utf-8')
idna.encode(d.lower(), strict=True, std3_rules=True)