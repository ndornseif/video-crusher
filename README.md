# video-crusher

Low level video manipulation.  
Version: 0.0.7

## Description

video-crusher is used to convert videos into different forms that enable easier low level access to their bytes and bits.  
First, the video is deconstructed into images at a specified sample frame rate.  
These frames are saved in a frame directory and can then be modified in their resolution, color depth etc.  
video-crusher is able to recombine them into a new video if desired.  
Is is also possible to reduce the audio in bandwidth or bit depth.  

## Command Line Arguments

- inputfile (Required, Path)  
The video file to work on.
- outputfile (Required, Path)  
Name of the resulting video file and frame directory.
- --verbose (Optional, Flag)  
Print full debug information.
- -colors (Optional, Int)  
Reduces the amount of color levels per channel to a specified amount.  
Acceptable values: 2-256  
- -fps (Optional, Int)  
The framerate to sample the input video at.
Defaults to input frame rate.
- -cpace (Optional, Selection)  
Convert video to this color space. (RGB=24bit color, L=8bit grayscale, 1=1bit grayscale)  
Defaults to RGB.
- -crushwidth and -crushheight (Optional, Int)  
Image resolution video will be downsampled to.
- -crushfactor (Optional, Int)  
Alternative way of setting -crushwidth and -crushheight.  
Reduces input video resolution by a set factor.
- --upsample (Optional, Flag)  
Returns frames to a higher resolution after downsampling.  
This preserves sharp edges during video compression.
- -upsamplewidth and -upsampleheight (Optional, Int)  
Resolution to upsample frames to.  
Defaults to input video resolution.
- --rmframeimg (Optional, Flag)  
Remove the frame directory after video recombination.
- --novideo (Optional, Flag)  
Don't recombine video.
- --noaudio (Optional, Flag)  
Don't add audio back into output video.
- --overwrite (Optional, Flag)  
Overwrite files that already exist at the specified output location.
- -lowpass (Optional, Int)  
Low pass audio at specified frequency.
- -highpass (Optional, Int)  
High pass audio at specified frequency.
- -audiobits (Optional, Int)  
Reduce audio bit resolution to specified level.
- -falsecolor (Optional, Palette)  
Supply a false color palette file to be applied to the video

### False color palettes

A false color palette is a text file consisting of 256 newline seperated 24-bit color values.  
Example palettes are supplied in the palettes/ directory.  
When applying a palette, the image is first converted to grayscale, the grayscale value is then used to look up an RGB color in the palette.  

## Acknowledgments
The example palettes were made from [matplotlib](https://matplotlib.org/) colormaps.

## Dependencies

- [python-pillow](https://pillow.readthedocs.io/en/stable/)
- [opencv](https://docs.opencv.org/4.x/index.html)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)

## Other

Published under GPL-3.0 license.