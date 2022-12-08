import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Image from "next/image";

import preprocessed from "../../public/preprocessed.jpg"
import postprocessed from "../../public/postprocessed.jpg"


const Home: NextPage = () => {

    const images = `##---------------Sources-------------------------##
# Image Processing for GymRetro:  https://github.com/deepanshut041/Reinforcement-Learning 
##------------------------------------------------##

import numpy as np
import cv2 as cv
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

sys.path.append(os.path.abspath(project_dir + '/source/agents'))
sys.path.append(os.path.abspath(project_dir + '/source/datasets'))
sys.path.append(os.path.abspath(project_dir + '/source/interface'))
sys.path.append(os.path.abspath(project_dir + '/source/vision'))

from deeplab import *
from deeplab_dataset import *
from color import *
from segmentation_labels import *

def preprocess_frame(screen, seg_model=None):
    """Preprocess Image.
        
        Params
        ======
            screen (array): RGB Image
            TODO   
            THESE ARE HARDCODED NOW, but worth breaking out into new methods later
            exclude (tuple): Section to be croped (UP, RIGHT, DOWN, LEFT)
            output (int): Size of output image
            TODO
            BLUR?
    """

    if seg_model is not None:
        seg = seg_model
        screen = seg.segment(screen)

    else:
        # convert image to gray scale
        screen = cv.cvtColor(screen, cv.COLOR_RGB2GRAY)
    # Scale down resolution to 1/4, remove color dimension
    # So our network takes in a 56x80 matrix    
    # TODO 
    screen = cv.resize(screen.astype(float), (84, 84))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    #screen = np.ndarray.flatten(screen)       
    return screen


def stack_frame(stacked_frames, frame, is_new):
    """Stacking Frames.
        
        Params
        ======
            stacked_frames (array): Four Channel Stacked Frame
            frame: Preprocessed Frame to be added
            is_new: Is the state First
        """
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames
    
# Overlays pixels of an image src2 onto image src1.
# Both images must be of the same size.
# Which pixels of scr2 that get copied are determined by mask
# For each pixels mask that equals 255, that cooresponding pixel of src2 is copied onto a pixel of src1
# 
# ex:
#	src1		src2		mask
#	a b c		1 2 3		0   0   255
#	d e f		4 5 6		255 255 0
#
#	result:
#	a b 3
#	4 5 f	
# 
# bot		bottom image
# top		top image (which will be overlayed onto src1)
# mask	which pixels are to be copied from top to bot
# return	image overlay of top and bot
# mask determines which top pixels will be placed over the bottom pixels.
def overlay_images(bot:np.ndarray, top:np.ndarray, mask:np.ndarray) -> np.ndarray:
    top = cv.bitwise_and(top, top, mask=mask)		# cut sillouette of top image
        
    mask = cv.bitwise_not(mask)						# invert
        
    bot = cv.bitwise_and(bot, bot, mask=mask)		# cut sillouette of bottom image
    
    img = cv.add(bot, top)
    
    return img

# Returns a mask with all pixels of shade color labeled as true (255)
#	and all other pixels labeled as false (0)
def mask_by_color(img:np.ndarray, color:Color, threshold=3) -> np.ndarray:
    
    # slice original image by color components
    img_b = img[:, :, 0]	# blue pixel components
    img_g = img[:, :, 1]	# green pixel components
    img_r = img[:, :, 2]	# red pixel components
            
    # Which pixels are part of the background (Which pixels should be made transparent)?
    _, lower_mask_b = cv.threshold(img_b, color.blue-threshold, 255, cv.THRESH_BINARY)
    _, upper_mask_b = cv.threshold(img_b, color.blue+threshold, 255, cv.THRESH_BINARY)
            
    _, lower_mask_g = cv.threshold(img_g, color.green-threshold, 255, cv.THRESH_BINARY)
    _, upper_mask_g = cv.threshold(img_g, color.green+threshold, 255, cv.THRESH_BINARY)
            
    _, lower_mask_r = cv.threshold(img_r, color.red-threshold, 255, cv.THRESH_BINARY)
    _, upper_mask_r = cv.threshold(img_r, color.red+threshold, 255, cv.THRESH_BINARY)
            
    mask_b = cv.bitwise_xor(lower_mask_b, upper_mask_b)
    mask_g = cv.bitwise_xor(lower_mask_g, upper_mask_g)
    mask_r = cv.bitwise_xor(lower_mask_r, upper_mask_r)
            
    # --- Finalize our Background and Foreground Masks ---
    mask = cv.bitwise_and(mask_b, mask_g)
    mask = cv.bitwise_and(mask, mask_r)
    
    return mask
        
def mask_by_intensity(img:np.ndarray, intensity:int) -> np.ndarray:
    # Which pixels are part of the background (Which pixels should be made transparent)?
    _, lower_mask = cv.threshold(img, intensity-1, 255, cv.THRESH_BINARY)
    _, upper_mask = cv.threshold(img, intensity, 255, cv.THRESH_BINARY)
            
    mask = cv.bitwise_xor(lower_mask, upper_mask)
    
    return mask

def draw_legend(img:np.ndarray) -> np.ndarray:
    legend = np.zeros((100, 65, 3), dtype=np.uint8)

    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    thickness = 1
    spacing = 12

    row = 10

    legend = cv.putText(
        img=legend,
        text="bg1",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.BACKGROUND1_COLOR.toTuple(),
    )

    row += spacing

    legend = cv.putText(
        img=legend,
        text="bg2",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.BACKGROUND2_COLOR.toTuple(),
    )
    
    row += spacing

    legend = cv.putText(
        img=legend,
        text="stage",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.STAGE_COLOR.toTuple(),	
    )
    
    row += spacing

    legend = cv.putText(
        img=legend,
        text="sonic",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.SONIC_COLOR.toTuple(),	
    )

    row += spacing

    legend = cv.putText(
        img=legend,
        text="robots",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.ROBOTS_COLOR.toTuple(),	
    )

    row += spacing
    
    legend = cv.putText(
        img=legend,
        text="items",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.ITEMS_COLOR.toTuple(),	
    )
    
    row += spacing

    legend = cv.putText(
        img=legend,
        text="hazards",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.HAZARDS_COLOR.toTuple(),	
    )
    
    row += spacing

    legend = cv.putText(
        img=legend,
        text="mechanical",
        org=(2, row),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color = SegmentationLabels.MECHANICAL_COLOR.toTuple(),	
    )

    row = 10
    col = 10
    rows = legend.shape[0]
    cols = legend.shape[1]

    img[row:row+rows, col:col+cols, :] = legend

    return img`

  return (
    <>
      <Head>
        <title>Basic Image Processing</title>
        <meta name="description" content="In order to optimize the training efficiency and accuracy of our Deep Q Learning Model, the team preprocessed images to reduce the dimensionality of input images" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Basic Image Processing</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center mb-3">In order to optimize the training efficiency and accuracy of our Deep Q Learning Model, the team preprocessed images to reduce the dimensionality of input images (prior to the development of our semantic segmentation model used in the final implementation). Native images from the OpenAI Retro emulator are 3-channel RGB images that measure 224 x 340 pixels. The team used methods from the openCV library to reduce the images to a single grayscale channel and reduce the image size to 84 x 84 pixels.</p>
            <p className="text-center">Images were converted to grayscale as it was hypothesized that we would not lose significant information by removing color. The images were downsampled to 84 x 84 pixels, utilizing bilinear interpolation. This image size was used as it is the native Atari image size, and much of the research behind our implementations stems from development of RL models using Atari platforms.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={preprocessed} className="max-h-96 w-auto p-6"></Image>
                <Image alt="Postprocessed sonic iamge" src={postprocessed} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/generalization"}>DeepQ Generalization -&gt;</Link>
          </button>
        </article>
        <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6 text-center">Our Code:</h3>
            <p>/source/vision/image_processing.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {images} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
