#!/usr/bin/env python
"""
color-palette-tools
Tools supplied with video-crusher to work with video-crusher palettes (.vcpal)
A palette is a text file consisting of 256 newline seperated 24-bit color values.
Palettes are encoded as ascii.
Example palettes are supplied in the palettes/ directory.
When applying a palette, the image is first converted to grayscale,
the grayscale value is then used to look up an RGB color in the palette.
All example palettes use the .vcpal filename extension.
"""

import matplotlib


def convert_color_format(input_color: tuple[float, float, float, float]) -> int:
    """
    Converts a color form a tuple to integer
    tuple four floats between 0 and 1 gets converted
    to a int between 0x000000 and 0xFFFFFF
    @params
        input_color - Required  : The color to convert (Tuple(Float, Float, Float))
    """
    return_color = int(input_color[2] * 255)
    return_color += int(input_color[1] * 255) << 8
    return_color += int(input_color[0] * 255) << 16

    return return_color


def generate_mpl_palette(colormap_name: str, levels: int = 256) -> list:
    """
    Converts a mathplotlib colormap into a palette
    A mount of distinctive colors in palette is set by the levels value
    @params
        colormap_name   - Required  : Name of mathplotlib colormap to convert
        levels          - Optional  : Number of distinctive colors (Int)
    """
    cmap = matplotlib.colormaps[colormap_name]
    pal = [None] * 256
    for index in range(256):
        div = 0xFF / levels
        mul = 0xFF / (levels - 1)
        pal[index] = convert_color_format(cmap(int((index // div) * mul)))

    return pal


def read_palette_file(file_path: str) -> list:
    """
    Read in a color pallete from a file.
    Palettes are saved as 256 24bit color values seperated by newlines
    @params
        file_path   - Required  : Path to palette file (Str)
    """
    colors = []
    with open(file_path, 'r', encoding='ascii') as file:
        for line in file:
            line = int(line.strip())
            colors.append(line)

    palette_lenth = len(colors)
    if palette_lenth != 256:
        raise ValueError(
            f'Palette is {palette_lenth} entries long. Exactly 256 required.')

    return colors


def save_palette_file(file_path: str, palette: list) -> None:
    """
    Save a color pallete to a file.
    Palettes are saved as 256 24bit color values seperated by newlines
    @params
        file_path   - Required  : Path to palette file (Str)
        palette     - Required  : Palette to save (List)
    """
    palette_lenth = len(palette)
    if palette_lenth != 256:
        raise ValueError(
            f'Palette is {palette_lenth} entries long. Exactly 256 required.')

    with open(file_path, 'w', encoding='ascii') as file:
        for color in palette:
            file.write(f'{color}\n')


if __name__ == "__main__":
    for colormap in list(matplotlib.colormaps):
        save_palette_file(
            f'palettes/{colormap}.vcpal',
            generate_mpl_palette(colormap))
