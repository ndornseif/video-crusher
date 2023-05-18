#!/usr/bin/env python
"""
color-palette-tools
Tools supplied with video-crusher to work with video-crusher palettes (.vcpal)
"""

import matplotlib

def convert_color_format(input_color: tuple[float, float, float, float]) -> int:
	"""
	Converts a color form a tuple of four floats between 0 and 1 to a value between 0x000000 and 0xFFFFFF
	@params
		input_color	- Required	: The color to convert (Tuple(Float, Float, Float))
	"""
	return_color = int(input_color[2] * 255)
	return_color += int(input_color[1] * 255) << 8
	return_color += int(input_color[0] * 255) << 16
	
	return return_color

def generate_mpl_palette(colormap_name: str, levels: int = 256) -> list:
	"""
	Converts a mathplotlib colormap into a 256 entry long conversion table containing 24bit RGB colors
	The amount of distinctive colors generated in the palette is set by the levels value
	@params
		colormap_name	- Required	: The name of the mathplotlib colormap to convert
		levels 			- Optional	: How many distinctive color levels to use (Int)
	"""
	cmap = matplotlib.colormaps[colormap_name]
	pal = [None] * 256
	for x in range(256):
		div = 0xFF / levels
		mul = 0xFF / (levels-1)
		pal[x] = convert_color_format(cmap(int((x // div) * mul)))

	return pal

def generate_eight_color_palette() -> list:
	"""
	Maps the eight basic colors achievable by using one bit per channel onto a palette
	"""
	pal = []
	for c in range(8):
		co = ((c & 1) * 0x0000FF)
		co += ((c & 2) * 0x007f80)
		co += ((c & 4) * 0x3fc000)
		pal.append(co)

	#Evenly map the eight colors onto the 256 palette entries
	pal = [pal[int(i/256.*8)] for i in range(256)]
	return pal


def read_palette_file(file_path: str) -> list:
	"""
	Read in a color pallete from a file.
	Palettes are saved as 256 24bit color values seperated by newlines
	@params
		file_path 	- Required	: Path to palette file (Str)
	"""
	colors = []
	with open(file_path) as f:
		for line in f: 
			line = int(line.strip()) 
			colors.append(line)

	if (len(colors) != 256):
		raise ValueError('Palette is {} entries long. Exactly 256 required.'.format(len(colors)))

	return colors

def save_palette_file(file_path: str, palette: list) -> None:
	"""
	Save a color pallete to a file.
	Palettes are saved as 256 24bit color values seperated by newlines
	@params
		file_path 	- Required	: Path to palette file (Str)
		palette		- Required	: 256 entry long conversion table containing 24bit RGB colors (List)
	"""
	if (len(palette) != 256):
		raise ValueError('Palette is {} entries long. Exactly 256 required.'.format(len(palette)))

	with open(file_path,'w') as f:
		for color in palette:
			f.write('{}\n'.format(color))



if __name__ == "__main__":
	for colormap in list(matplotlib.colormaps):
		save_palette_file('palettes/{}.vcpal'.format(colormap), generate_mpl_palette(colormap))
