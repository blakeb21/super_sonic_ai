import type { NextPage } from "next";
import Head from "next/head";
import Link from "next/link";
import Header from "../components/header";
import Image from "next/image";

import preprocessed from "../../public/preprocessed2.jpg"
import postprocessed from "../../public/postprocessed2.jpg"


const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>Semantic Segmentation</title>
        <meta name="description" content="Exploring how to advance our model and developing NEAT and DeepQ agents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Phase 4: Semantic Segmentation</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Introduction</h2>
            <p className="text-center">The input space of our decision making model is huge. In our project, we use a variety of neural network based decision making models like NEAT and DeepQ. There are 7 possible decisions which our models can make: move right, move left, jump, stand still, crouch, move right and jump, move left and jump. These models “look” at the game&apos;s screen and try to make meaningful decisions based on what it “sees” in order to win the game. Initially in our project, we input the entire image to the neural networks meaning our neural networks will base their decisions on every pixel they see. The input image is a 224x340 3 channel RGB image. This is 228,480 values which will be input to our neural network. Although more input values improves the overall capacity of a neural network&apos;s performance, it would take a larger neural network to process it. The problem is that a slightly larger neural network takes significantly longer to train especially in reinforcement learning. Reinforcement learning allows us to train a neural network without training data. But reinforcement learning comes at a greater cost, longer training times. Each added node to the neural network could increase its training time exponentially. With an unlimited amount of resources, a neural network of any size could be trained to its full potential, but this could require months or even years of training. This describes the “Curse of Dimensionality” problem. The solution is Dimensionality Reduction where we reduce the number of inputs to the neural network without losing a significant amount of useful information. </p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">One approach (shrinking the image)</h2>
            <p className="text-center">One form of dimensionality reduction is to simply shrink the image. We also convert it from color to grayscale, cutting the size of the image in thirds. But this method removes a significant amount of useful information that the decision making models need. By simply reducing the size of the image by fourths and removing color, the image looks noisy and objects are unrecognizable even to a human player. Background pixels get mixed with foreground pixels making it very difficult to make decisions even for a human.</p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Semantic Segmentation with Neural Networks and how it works</h2>
            <p className="text-center mb-3">The game&apos;s output image is big. Fortunately, most of its pixels are not significantly useful for a decision making model like DeepQ or Neat. Most of what shows up on the screen is artwork that makes the game more appealing to human players but not very useful for deciding what moves to make. As we play the game as humans, our brains naturally simplify what we see into segments. At first, we see the game pixel by pixel. We assume that all pixels are equally important to winning the game. As we play we begin to realize which pixels are part of which entities. At the same time, we learn how each entity behaves. Some pixels are part of the ground. That&apos;s what we can walk on. Some pixels are part of the background artwork. We can ignore these because they don&apos;t do anything. Some pixels are part of good items that help us win the level. Others are part of hazardous objects that we want to avoid. Our brains naturally find these patterns like magic.</p>
            <p className="text-center">It turns out that computers can also do this using machine learning. Any large enough neural network, even a reinforcement learning model like DeepQ or Neat can perform image segmentation on their own even without the help of explicit image segmentation. It can use the original image and with enough training discover which pixels are part of what entities and how the entities behave. This requires a larger neural network and lots more training time. To overcome this we create another neural network which focuses solely on image segmentation. This neural network can be smaller and can be trained faster and more efficiently with supervised learning. The output of this segmentation neural network will be the classification of each pixel. This identifies which pixels are part of which entities. If a pixel is classified as 0, it is part of the 1st background. These can be ignored. If a pixel is classified as 1, it is part of the 2nd background, these pixels make up walls and the underground. We can walk through these. Class 2 are the pixels that we can walk on. Classification 5 make up robots that we have to destroy. The output of the Image Segmentation Model will look something like this.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={preprocessed} className="max-h-96 w-auto p-6"></Image>
                <Image alt="Postprocessed sonic iamge" src={postprocessed} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center mb-3">The left image is what is input to the image segmentation neural network. The image on the right is the output. As you can see, the artwork and background are cleared away. And the foreground objects are simplified into just what the decision making model needs and nothing else. This segmented image is passed to a decision making model like DeepQ or NEAT. With this image, the decision making model has a simplified input containing only what it needs inorder to make accurate and meaningful decisions. The segmented image may not be as appealing to a human, but this is how parts of our brains see the world anyway. </p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Dimensionary Reduction</h2>
            <p className="text-center">In terms of the number of inputs to the decision making model, we are still inputting the same number of pixels because the size of the segmented image is still the same. But because the pixels are segmented we can shrink the image without losing as much useful information. The benefit of image segmentation is that we can simplify an image, reduce noise, classify each pixel and shrink the image into a smaller one without losing significant information. This makes it easier for a decision making model to process the image. We can reduce the size of the decision making neural network resulting in faster training even with reinforcement learning.</p>
          </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/dataset_creation"}>Dataset Creation -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Home;
