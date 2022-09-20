import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";


const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>SuperSonicAI</title>
        <meta name="description" content="Welcome to the Super Sonic AI project." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col items-center justify-center  p-4">
            <h1 className="text-purple-500 text-2xl">Welcome to SuperSonicAI</h1>
            <p className="">Content coming soon!</p>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
