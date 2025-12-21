from PIL import Image
import struct
import ctypes

for i in range(126):
  input_name = str(i) + '.data'
  output_name = str(i) + '.png'
  print(input_name)
  fin = open(input_name, 'rb')
  (w, h) = struct.unpack('ii', fin.read(8))
  buff = ctypes.create_string_buffer(4 * w * h)
  fin.readinto(buff)
  fin.close()
  img = Image.new('RGBA', (w, h))
  pix = img.load()
  offset = 0
  for j in range(h):
    for i in range(w):
      (r, g, b, a) = struct.unpack_from('cccc', buff, offset)
      pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
      offset += 4
  print(output_name)
  img.save(output_name)
  