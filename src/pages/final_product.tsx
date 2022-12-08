import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';

const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>DeepQ with Segmentation</title>
        <meta name="description" content="Exploring how to advance our model and developing NEAT and DeepQ agents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Phase 5: DeepQ with Segmentation</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Introduction</h2>
            <p className="text-center">This experimentation discussed in the preceding phases led us to a solution that can finish the first level of Sonic and generalizes fairly well to unseen environments. This solution involves generating a synthetic dataset of images from the game to train a DeepLab V3 semantic segmentation model. We then apply this trained model to the Sonic emulator as a preprocessing step to feed segmented images into a Deep Q Learning Agent. The Deep Q Learning Agent was then on several levels of Sonic to develop the optimal policy of state-action pairs to support generalization on unseen environments. </p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">DeepQ with Image Segmentation</h2>
            <p className="text-center">Once the Image Segmentation Model is trained, it is added to the beginning of the DeepQ Model. During inference, the game image is passed through the Image Segmentation Model which classifies each pixel and outputs the segmented image. The output image has the same width and height as the input image put only has 1 channel. This channel contains the classifications of each pixel. This image is easier to make decisions on because the image is simplified into what each pixel is. Redundant and noisy pixels are cleaned up so that DeepQ only needs to see what&apos;s relevant for making decisions. But this only mitigates the “Curse of Dimensionality” slightly. The segmented image is still large and therefore requires a smaller but still large DeepQ model to extract meaningful information necessary for decision making. The real benefit of image segmentation is that the image can be made smaller without losing significant information. Most adjacent pixels are the same anyway, so shrinking the image will result in an image very similar to the original. This results in a smaller DeepQ model with fewer nodes and fewer layers. This smaller DeepQ model can be trained many times faster without reducing the total model&apos;s overall potential. </p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
                <video autoPlay muted loop className="max-h-96">         
                    <source src="/final.mp4" type="video/mp4"/>       
                </video>
            </div>
        </article>
      </main>
    </>
  );
};

export default Home;
