#!/usr/bin/env python
"""
video-crusher
Utility used to make videos worse for artistic or technical reasons.

TODO:
Replace usage of PIL with CV2
"""

import logging
import shutil
import os
import sys
import argparse
from argparse import Namespace
from enum import Enum
from dataclasses import dataclass

import PIL
from PIL import Image
import cv2
import ffmpeg

from color_palette_tools import read_palette_file


__author__ = "N. Dornseif"
__copyright__ = "Copyright 2023, N. Dornseif"
__license__ = "GNU General Public License v3.0"
__version__ = "0.1.0"


class Colorspace(Enum):
    """
    Enum used to track different suported colorspaces
    """
    SINGLE_BIT = '1'  # 1 bit color
    GRAYSCALE = 'L'   # 8 bit grayscale
    RGB_COLOR = 'RGB'  # 3x8 bit full color


def print_progress_bar(iteration: int, total: int, length: int = 100, fill: str = '█', print_end: str = "\r") -> None:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ('{0:.1f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r|{progress_bar}| {percent}%', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def reduce_image_color_depth(input_image: PIL.Image, levels:int) -> PIL.Image:
    """
    Combines all color values in a channel into (levels) different values
    @params:
        input_image - Required  : Image to work on (PIL.Image)
        levels      - Required  : How many different values per color channel to use (Int)
    """
    logger = logging.getLogger(__name__)

    if input_image.mode == Colorspace.SINGLE_BIT.value:
        logger.debug(
            'Image is single bit color.'
            'Reducing color depth is not possible. Skipping.')
        return input_image

    div = 0xFF / levels
    mul = 0xFF / (levels - 1)
    return input_image.point(lambda x: int((x // div) * mul))


def apply_colors_to_grayscale(input_image: PIL.Image, palette: list) -> PIL.Image:
    """
    Takes a grayscale image and converts it back to rgb
    Accomplished by applying a false color palette
    @params
        input_image - Required  : Image to work on (PIL.Image)
        palette     - Required  : Color palette (List)
    """
    logger = logging.getLogger(__name__)
    logger.debug('Applying palette to image.')
    if input_image.mode == Colorspace.RGB_COLOR.value:
        logger.debug('Image is not grayscale. Converting.')
        input_image = PIL.ImageOps.grayscale(input_image)

    # Expand image back out to 24 bits per pixel
    # Split into three color bands.
    band_image = input_image.convert('RGB').split()

    for band in range(0, 3):
        # Run method once for each of the color channels
        # Take the 24 bit value defined by the palette
        # Extract one of the color channels by downshifting
        # Downshift by 16 bits for Red, 8 for Green and 0 for Blue.
        # Bitwise AND with 0xFF sets all bits except the last eight to zero
        logger.debug('Processing color band %s.', band)

        band_image[band].paste(band_image[band].point(
            lambda x: (palette[x] >> (16 - (band * 8))) & 0xFF))

    logger.debug('Palette successfully applied to image.')
    return Image.merge(Colorspace.RGB_COLOR.value, band_image)


def configure_logger() -> None:
    """
    Configures a logger accessible using __name__ as its identifier
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def cli_parse() -> Namespace:
    """
    Generates argument parser and parses CLI args
    Returnins them as a Namespace object
    """
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog='video-crusher',
        description='Utility used to make videos worse')

    parser.add_argument('inputfile',
                        help='the video file to process')

    parser.add_argument('outputfile',
                        help='where to save the resulting video. \
                        Individual frames will also be saved here')

    parser.add_argument('--version',
                        action='version', version='%(prog)s ' + __version__)

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='print extensive debug information')

    parser.add_argument('-colors',
                        default=256, type=int,
                        help='how many different color levels to retain \
                        Applies to each color channel individualy.')

    parser.add_argument('-fps',
                        type=float,
                        help='framerate to save and process frames at. \
                        Defaults to input video framerate')

    parser.add_argument('-cspace',
                        choices=['RGB', 'L', '1'],
                        help='colorspace of the resulting video. \
                        (RGB=24bit color, L=8bit grayscale, 1=1bit grayscale)')

    parser.add_argument('-crushwidth',
                        type=int,
                        help='image width to downsample to. \
                        (Optional if crush factor specified)')

    parser.add_argument('-crushheight',
                        type=int,
                        help='image height to downsample to. \
                        (Optional if crush factor specified)')

    parser.add_argument('-crushfactor',
                        type=int,
                        help='reduce resolution by this factor. \
                        (Optional if crush width and height specified)')

    parser.add_argument('--upsample',
                        action='store_true',
                        help='return images to original or \
                        specified resolution after downsampling. \
                        This preserves sharp pixel edges')

    parser.add_argument('-upsamplewidth',
                        type=int,
                        help='image width to upsample to. \
                        Defaults to input video dimensions')

    parser.add_argument('-upsampleheight',
                        type=int,
                        help='image height to upsample to. \
                        Defaults to input video dimensions')

    parser.add_argument('--rmframeimg',
                        action='store_true',
                        help='remove individual frame images \
                        after processing and preserve only output video')

    parser.add_argument('--novideo',
                        action='store_true',
                        help='dont combine frames into video after processing')

    parser.add_argument('--noaudio',
                        action='store_true',
                        help='dont add original audio back \
                        onto video after processing')

    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrites if output files already exist')

    parser.add_argument('-lowpass',
                        default=20_000, type=int,
                        help='frequency to low pass filter the audio at')

    parser.add_argument('-audiobits',
                        type=int,
                        help='reduces audio bit resolution to set level')

    parser.add_argument('-highpass',
                        default=1, type=int,
                        help='frequency to high pass filter the audio at')

    parser.add_argument('-falsecolor',
                        help='supply a false color palette file \
                        to be applied to the video')

    parsed_args = parser.parse_args()

    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.debug('Argparse finished.')

    return parsed_args


@dataclass
class CrushConfig:
    """
    A dataclass to manage the parameters of a video crusher
    """
    input_file_path: str
    output_file_path: str
    input_video_file: cv2.VideoCapture = None
    frame_dir: str = None
    verbose: bool = False
    crush_width: int = None
    crush_height: int = None
    crush_video: bool = True
    color_levels: int = 256
    upsample: bool = None
    upsample_width: int = None
    upsample_height: int = None
    output_fps: float = None
    colorspace: Colorspace = Colorspace.RGB_COLOR
    falsecolor: bool = False
    falsecolor_palette: list = None
    original_width: int = None
    original_height: int = None
    original_fps: int = None
    audio_low_pass: int = None
    audio_high_pass: int = None
    audio_file_path: str = None
    reduce_audio_bit_depth: bool = False
    audio_bit_depth: int = None
    temp_video_path: str = None
    remove_frame_images: bool = False
    combine_frames: bool = True
    add_audio: bool = True


def cli_args_to_crushconfig(cli_args: Namespace) -> CrushConfig:
    """
    Takes in a Namespace object of passed CLI args and creates a crusher config
    It also loads the input video file in the process
    @params
        cli_args        - Required  : CLI args returned by argparse (Namespace)
    """
    logger = logging.getLogger(__name__)
    crush_config = CrushConfig(
        input_file_path=cli_args.inputfile,
        output_file_path=cli_args.outputfile)

    logger.debug('Loading input video at: %s', crush_config.input_file_path)

    crush_config.verbose = cli_args.verbose

    if not os.path.isfile(crush_config.input_file_path):
        raise FileNotFoundError(
            f'Input video file at {crush_config.input_file_path} not found.')

    if (os.path.exists(crush_config.output_file_path)
            and not cli_args.overwrite):
        raise FileExistsError(
            'Output file at'
            + crush_config.output_file_path
            + 'already exists. Use --overwrite to save anyways.')

    crush_config.frame_dir = os.path.splitext(crush_config.output_file_path)[0]

    if os.path.exists(crush_config.frame_dir) and (not cli_args.overwrite):
        raise FileExistsError(
            'Frame save directory at'
            + crush_config.frame_dir
            + 'already exists. Use --overwrite to save anyways.')

    crush_config.input_video_file = cv2.VideoCapture(
        crush_config.input_file_path)

    crush_config.original_width = int(
        crush_config.input_video_file.get(cv2.CAP_PROP_FRAME_WIDTH))

    crush_config.original_height = int(
        crush_config.input_video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crush_config.original_fps = int(
        crush_config.input_video_file.get(cv2.CAP_PROP_FPS))

    logger.debug('Input video properties: %sx%s @ %sfps',
                 crush_config.original_width,
                 crush_config.original_height,
                 crush_config.original_fps)

    if (crush_config.original_width == 0
            or crush_config.original_height == 0
            or crush_config.original_fps == 0):
        raise ValueError('Input video could not be read.')

    if cli_args.crushfactor is not None:
        logger.debug('Crush factor %s specified.', cli_args.crushfactor)
        crush_config.crush_width = int(
            crush_config.original_width / cli_args.crushfactor)

        crush_config.crush_height = int(
            crush_config.original_height / cli_args.crushfactor)

        logger.debug('Crush resolution set to: %sx%s',
                     crush_config.crush_width, crush_config.crush_height)

        crush_config.crush_video = True

    elif (cli_args.crushwidth is not None
            and cli_args.crushheight is not None):
        logger.debug('Crush resolution manually specified.')
        crush_config.crush_width = cli_args.crushwidth
        crush_config.crush_height = cli_args.crushheight
        logger.debug('Crush resolution set to: %sx%s',
                     crush_config.crush_width, crush_config.crush_height)

        crush_config.crush_video = True

    else:

        logger.debug(
            'No proper crush resolution specified. Not crushing video.')
        crush_config.crush_video = False

    if (cli_args.colors < 2) or (cli_args.colors > 256):
        raise ValueError(
            f'Invalid color depth {cli_args.colors} specified.'
            'Range 2-256 expected.')

    crush_config.color_levels = cli_args.colors

    if cli_args.cspace is not None:
        logger.debug('Colorspace %s specified.', cli_args.cspace)
        crush_config.colorspace = Colorspace(cli_args.cspace)
    else:

        logger.debug('No colorspace specified. Defaulting to RGB.')
        crush_config.colorspace = Colorspace.RGB_COLOR

    crush_config.upsample = cli_args.upsample

    if crush_config.upsample:
        logger.debug('Upsampling enabled.')
        if cli_args.upsamplewidth is not None:
            crush_config.upsample_width = cli_args.upsamplewidth
            logger.debug('Upsample width specified as %s',
                         cli_args.upsamplewidth)

        else:

            logger.debug(
                'No upsample width specified. Defaulting to source res. (%s)',
                crush_config.original_width)

            crush_config.upsample_width = crush_config.original_width

        if cli_args.upsampleheight is not None:
            logger.debug('Upsample height specified as %s',
                         cli_args.upsampleheight)

            crush_config.upsample_height = cli_args.upsampleheight
        else:

            logger.debug(
                'No upsample height specified. Defaulting to source res. (%s)',
                crush_config.original_height)

            crush_config.upsample_height = crush_config.original_height

    if cli_args.fps is not None:
        if cli_args.fps > crush_config.original_fps:
            logger.warning(
                'User specified sample fps (%s) higher than source fps (%s)',
                cli_args.fps, crush_config.original_fps)

            logger.warning('Smaple fps set to source fps.')
            crush_config.output_fps = crush_config.original_fps
        else:

            logger.debug('Sample fps set to user specified value of %s',
                         cli_args.fps)

            crush_config.output_fps = cli_args.fps
    else:

        logger.debug('No sample fps specified. Setting to source fps: %s.',
                     crush_config.original_fps)

        crush_config.output_fps = crush_config.original_fps

    if cli_args.falsecolor is not None:
        if not os.path.isfile(cli_args.falsecolor):
            raise FileNotFoundError(
                f'Palette file at {cli_args.falsecolor} not found.')

        crush_config.falsecolor = True
        crush_config.falsecolor_palette = read_palette_file(
            cli_args.falsecolor)

        logger.debug('Falsecolor palette specified at %s Enabling falsecolor.',
                     cli_args.falsecolor)

    else:
        logger.debug('No falsecolor palette specified. Disabling falsecolor.')
        crush_config.falsecolor = False

    crush_config.audio_low_pass = cli_args.lowpass
    crush_config.audio_high_pass = cli_args.highpass
    logger.debug('Audio filters configured. HP:%s LP:%s',
                 crush_config.audio_high_pass, crush_config.audio_low_pass)

    if cli_args.audiobits is not None:
        logger.debug('Reduced audio bit depth of %s configured.',
                     cli_args.audiobits)

        crush_config.audio_bit_depth = cli_args.audiobits
        crush_config.reduce_audio_bit_depth = True

    crush_config.remove_frame_images = cli_args.rmframeimg
    crush_config.combine_frames = not cli_args.novideo
    crush_config.add_audio = not cli_args.noaudio

    crush_config.audio_file_path = os.path.join(
        crush_config.frame_dir, 'audio.mp3')

    crush_config.temp_video_path = os.path.join(
        crush_config.frame_dir, 'temp.mp4')

    logger.debug('CLI arg processing finished.')
    return crush_config


class Crusher():
    """
    Manages all funtions and variables pertaining to a single input video
    """

    def __init__(self, crush_config: CrushConfig, logger: logging.Logger = None) -> None:
        """
        Initialize a Crusher object.
        @params
            self        - Required  : Crusher class instance
            crush_config- Required  : Configuration dataclass (CrushConfig)
        """
        self.config = crush_config

        if os.path.isdir(self.config.frame_dir):
            shutil.rmtree(self.config.frame_dir)

        os.mkdir(self.config.frame_dir)

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process_video_to_disk(self) -> None:
        """
        Takes the video defined in Crusher.config.input_video_file
        then saves its contents as individual bitmap images
        The images are saved as {n}.ppm in Crusher.config.frame_dir
        Also applies transformations to all frames
        The Transformations are configured using the Crusher.config variables
        @params
            self        - Required  : Crusher class instance
        """
        self.logger.debug('Deconstructing video to disk at: %s',
                          self.config.frame_dir)

        result_framestep = int(round(
            self.config.original_fps / self.config.output_fps))

        self.logger.debug('Saving every %s frames.', result_framestep)
        save_counter = 0
        total_frame_count = int(
            self.config.input_video_file.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_counter in range(0, total_frame_count):
            print_progress_bar(frame_counter + 1, total_frame_count)
            state, frame = self.config.input_video_file.read()
            if not state:
                self.logger.warning('Frame read failed! Frame:%s',
                                    frame_counter)
                break

            if not frame_counter % result_framestep:
                frame_path = os.path.join(
                    self.config.frame_dir,
                    str(save_counter) + '.ppm')

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                PIL_frame = Image.fromarray(frame)
                PIL_frame = self.process_single_frame(PIL_frame)
                PIL_frame.save(frame_path)

                save_counter += 1
                if not state:
                    self.logger.warning('Frame write failed! Frame:%s',
                                        frame_counter)
                    break

    def process_single_frame(self, frame: PIL.Image) -> PIL.Image:
        """
        Applies transformations to a frame
        The Transformations are configured using the Crusher.config variables
        @params
            self        - Required  : Crusher class instance
            frame       - Required  : The image to work on
        """
        frame = frame.convert(self.config.colorspace.value)
        if self.config.crush_video:
            frame = frame.resize(
                (self.config.crush_width, self.config.crush_height),
                Image.Resampling.LANCZOS)

        if self.config.colorspace.value != Colorspace.SINGLE_BIT.value:
            frame = reduce_image_color_depth(
                frame, self.config.color_levels)

        if self.config.falsecolor:
            frame = apply_colors_to_grayscale(
                frame, self.config.falsecolor_palette)

        if self.config.upsample:
            frame = frame.resize(
                (self.config.upsample_width, self.config.upsample_height),
                Image.Resampling.NEAREST)

        return frame

    def export_audio_to_disk(self) -> None:
        """
        Takes the input file and saves its audio as mp3
        A high- and lowpass filter defined by Crusher.config are applied
        If Crusher.config.reduce_audio_bit_depth flag is set,
        reduces bitdepth to value specified by Crusher.config.audio_bit_depth
        Audio saved at location defined by Crusher.config.audio_file_path
        @params
            self        - Required  : Crusher class instance
        """
        self.logger.debug('Audio processing started.')
        input_file = ffmpeg.input(self.config.input_file_path)

        if self.config.audio_high_pass < 20_000:
            audio_stream = input_file.audio.filter_(
                'highpass', f=self.config.audio_high_pass)

        if self.config.audio_low_pass > 1:
            audio_stream = audio_stream.filter_(
                'lowpass', f=self.config.audio_low_pass)

        if self.config.reduce_audio_bit_depth:
            audio_stream = audio_stream.filter_(
                'acrusher', bits=self.config.audio_bit_depth)
        output_stream = ffmpeg.output(
            audio_stream,
            filename=self.config.audio_file_path)
        output_stream.run(quiet=not self.config.verbose)
        self.logger.debug('Audio processed and saved at: %s',
                          self.config.audio_file_path)

    def combine_frames_from_disk(self) -> None:
        """
        Takes all frames in the frame dir and combines them into a video file
        This video file is saved as defined by Crusher.config.temp_video_path
        @params
            self        - Required  : Crusher class instance
        """
        self.logger.debug('Combining frames to video.')
        frame_stream = ffmpeg.input(
            f'{self.config.frame_dir}/%d.ppm',
            framerate=self.config.output_fps)
        output_stream = ffmpeg.output(
            frame_stream,
            filename=self.config.temp_video_path,
            crf=15, preset='slow', movflags='faststart', pix_fmt='yuv420p')
        output_stream.run(quiet=not self.config.verbose)
        self.logger.debug('Frame images combined and saved at: %s',
                          self.config.temp_video_path)

    def stitch_audio_to_video(self) -> None:
        """
        Combines video file with audio.mp3 in the frame directory
        frame directory defined by Crusher.config.frame_dir
        Saves combined video as defined by Crusher.config.output_file_path
        @params
            self        - Required  : Crusher class instance
        """
        self.logger.debug('Stiching audio and video files.')
        input_audio = ffmpeg.input(self.config.audio_file_path)
        input_video = ffmpeg.input(self.config.temp_video_path)
        output_stream = ffmpeg.output(
            input_video.video, input_audio.audio,
            filename=self.config.output_file_path,
            shortest=None, vcodec='copy').overwrite_output()

        output_stream.run(quiet=not self.config.verbose)
        self.logger.debug('Audio and video combined and saved at: %s',
                          self.config.output_file_path)
        os.remove(self.config.temp_video_path)
        self.logger.debug('Removed temp video file at: %s',
                          self.config.temp_video_path)
        os.remove(self.config.audio_file_path)
        self.logger.debug('Removed temp audio file at: %s',
                          self.config.audio_file_path)

    def process_video(self) -> None:
        """
        Performs all steps necessary to crush video
        behavior determined by Crusher.config
        @params
            self        - Required  : Crusher class instance
        """
        if self.config.add_audio and self.config.combine_frames:
            self.logger.info('Processing audio.')
            self.export_audio_to_disk()

        self.logger.info('Processing frames.')
        self.process_video_to_disk()

        if self.config.combine_frames:
            self.logger.info('Combining frames.')
            self.combine_frames_from_disk()

            if self.config.add_audio:
                self.logger.info('Reapplying audio.')
                self.stitch_audio_to_video()
            else:
                shutil.move(self.config.temp_video_path,
                            self.config.output_file_path)

        if self.config.remove_frame_images:
            shutil.rmtree(self.config.frame_dir)

        video_name = os.path.splitext(self.config.output_file_path)[0]
        self.logger.info('Video %s finished processing.', video_name)


def main() -> int:
    """
    Parses CLI args and provides them to a Crusher object.
    """
    configure_logger()
    cli_args = cli_parse()
    c_config = cli_args_to_crushconfig(cli_args)
    my_crusher = Crusher(crush_config=c_config)
    my_crusher.process_video()
    return 0


if __name__ == "__main__":
    sys.exit(main())
