import binascii
import struct

no = 365.9731750488281


float_value = 1.5

float_bytes = struct.pack('f', float_value)

print(float_bytes)

int_value = struct.unpack('<f', float_bytes)[0]
print(int_value)