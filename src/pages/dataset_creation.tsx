import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import Image from "next/image";

import preprocessed from "../../public/preprocessed2.jpg"
import postprocessed from "../../public/postprocessed2.jpg"

import background from "../../public/background.jpg"
import level from "../../public/level.jpg"
import sprite from "../../public/sprite.jpg"
import images from "../../public/images.jpg"

const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>Dataset Development</title>
        <meta name="description" content="Exploring how to advance our model and developing NEAT and DeepQ agents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Dataset Creation</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center">We choose to make image segmentation a supervised learning problem meaning to train a segmentation model, we need a training dataset. There is no training dataset online for our specific problem, so we choose to create our own. A segmentation dataset is simple. All we need is a collection of images which come from the game along with their segmentations.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={preprocessed} width={574} height={370} className="max-h-96 w-auto p-6"></Image>
                <Image alt="Postprocessed sonic iamge" src={postprocessed} width={570} height={374} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center mb-3">The image on the left is the unsegmented image. The right is the segmented one. The legend is only there for human readability and specifies what classification each color represents. The segmentations are the same size as the original images, but instead of rgb pixels, each pixel stores an integer representing the classification of the corresponding pixel in the original. The original images can be copied straight from the game&apos;s screen. The challenge is labeling each pixel to get the segmented images. We can manually segment images but this process is not scalable as each image takes about 15 minutes to segment by hand. And we would need thousands of images to train a network sufficiently.</p>
            <p className="text-center">An alternative approach is to make a program that generates a dataset of synthetic segmented and unsegmented images automatically. We call this program the dataset generator. The unsegmented images will be constructed from images taken from the games Read Only Memory (ROM). These images are broken down into components: background, tilesets, sprites.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={background} width={508} height={504} className="max-h-96 w-auto p-6"></Image>
                <Image alt="Postprocessed sonic iamge" src={level} width={1170} height={580} className="max-h-96 w-auto p-6"></Image>
                <Image alt="Postprocessed sonic iamge" src={sprite} width={728} height={340} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center">We would manually segment these images by hand to create a foundation dataset. Once we have our foundation dataset, our dataset generator stitches them together similar to the way the game renders images on the screen. Our dataset generator will not generate images exactly like they would appear in the game. It only overlaps backgrounds, tilesets and sprites randomly to create a dataset that a neural network can learn from. With this method, we were able to create a synthetic dataset of any size in a short time. Below is an example of what the dataset generator creates.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={images} width={870} height={920} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/model_training"}>Segmentation Training -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Home;
