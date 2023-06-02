# video-crusher

Very lossy video compression.  

## Description

video-crusher is used to make videos worse for artistic or technical reasons.  
Videos get deconstructed into frames at a specified sample frame rate.  
These frames are saved in a frame directory and can then be modified in their resolution, color depth, etc.  
video-crusher is able to recombine them into a new video if desired.  
Is is also possible to reduce the audio bandwidth or audio bit depth.  

## Command Line Arguments

```
$ video-crusher INPUTFILE OUTPUTFILE
```
The video files to work on.  
OUTPUTFILE will also be used as the name for the frame directory.  


```
$ video-crusher --verbose
```  
Print full debug information.  


```
$ video-crusher -colors INT
```
Reduces the amount of color levels per channel to a specified amount.  
Acceptable values: 2 - 256  


```
$ video-crusher -fps INT
```
The framerate to sample the input video at.  
Defaults to input frame rate.  


```
$ video-crusher -cspace SELECTION
```
Convert video to this color space. (RGB=24-bit color, L=8-bit grayscale, 1=1-bit grayscale)  
Defaults to RGB.  


```
$ video-crusher -crushwidth INT -crushheight INT
```
Resolution video will be downsampled to.  


```
$ video-crusher -crushfactor INT
```  
Alternative way of setting -crushwidth and -crushheight.  
Reduces input video resolution by a set factor.  


```
$ video-crusher --upsample
```
Returns frames to a higher resolution after downsampling.  
This preserves sharp pixel edges during video compression.  


```
$ video-crusher -upsamplewidth INT -upsampleheight INT
``` 
Resolution to upsample frames to.     
Defaults to input video resolution.   


```
$ video-crusher --rmframeimg
```
Remove the frame directory after video recombination.  


```
$ video-crusher --novideo
```
Don't recombine video.  


```
$ video-crusher --noaudio
```
Don't add audio back into output video.  
Please set this flag if input video contains no audio.  


```
$ video-crusher --overwrite
```
Overwrite files that already exist at the specified output location.  


```
$ video-crusher -lowpass INT
```
Low pass audio at specified frequency.  


```
$ video-crusher -highpass INT
```
High pass audio at specified frequency.  


```
$ video-crusher -audiobits INT
```
Reduce audio bit resolution to specified level.  


```
$ video-crusher -falsecolor PALETTE_FILE
```
Supply a false color palette to be applied to the video.  

### False color palettes

A false color palette is a text file consisting of 256 newline seperated 24-bit color values.  
Example palettes are supplied in the palettes/ directory.  
Palettes are encoded as ascii.  
When applying a palette, the image is first converted to grayscale, the grayscale value is then used to look up an RGB color in the palette.  
All example palettes use the .vcpal filename extension.  

## Example outputs
Input video used for examples: [Noisestorm - Crab Rave (Official Music Video)](https://youtu.be/cE0wfjsybIQ)  
```
$ video-crusher in.mp4 out.mp4 -fps 10 -crushfactor 10 -cspace L -colors 4 --upsample -lowpass 500  
```
[Output video](https://youtu.be/iQYhlxVNbrg)  
```
$ video-crusher in.mp4 out.mp4 -fps 10 -crushfactor 4 -falsecolor palettes/nipy_spectral.vcpal --upsample -audiobits 1  
```
[Output video](https://youtu.be/iZBzmmg-jKQ)  
```
$ video-crusher in.mp4 out.mp4 -fps 10 -crushfactor 17 -cspace L -colors 2 --upsample  
```
[Output video](https://youtu.be/-ywP-9Joyqs)  
```
$ video-crusher in.mp4 out.mp4 -fps 10 -crushfactor 17 -cspace 1 --upsample  --noaudio
```
[Output video](https://youtu.be/DpW4GvWdQes)  

## Considerations

Decompressing high resolution or long videos to bitmaps takes a lot of drive space.

|    Color    | Resolution | Frame rate | Size         |
|-------------|------------|------------|--------------|
| 24-bit RGB  | 1920x1080  | 30 fps     | 10.4 GiB/min |
| 24-bit RGB  | 1920x1080  | 10 fps     | 3.5 GiB/min  |
| 24-bit RGB  | 720x480    | 30 fps     | 1.7 GiB/min  |
| 24-bit RGB  | 720x480    | 10 fps     | 580 MiB/min  |

## Acknowledgments

The example palettes were made from [matplotlib](https://matplotlib.org/) colormaps.

## Dependencies

- [python-pillow](https://pillow.readthedocs.io/en/stable/)
- [opencv](https://docs.opencv.org/4.x/index.html)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)

## Other

Published under GPL-3.0 license.