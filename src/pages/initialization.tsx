import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";

import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Initialization: NextPage = () => {

    const packages = "pip install <package>"

    const randomAgent = `import retro

def main():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record='.')
    obs = env.reset() # Reset button on emulator
    done = False
    for i in range(2000): 
            # Action space is array of the 12 buttons on genesis controller. 
            # 1 if pressed, 0 if not
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        env.render() #Show emulator

    env.close()

if __name__ == "__main__":
    main()`

  return (
    <>
      <Head>
        <title>Initialization</title>
        <meta name="description" content="Get started with our repository and run the code for yourself." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Phase 1: Initialization</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Introduction</h2>
            <p className="text-center">SuperSonicAI has lots of software dependencies. It requires a modern operating system like Windows, Mac or Linux. It needs python, pip, gym-retro, pytorch, numpy, opencv, and a few other packages that can be installed using pip. The last requirement is the game&apos;s read only memory provided in the git repo. Also, these instructions expect users to have some knowledge on using the command line.</p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Installing git</h2>
            <p className="text-center">The first step in setting up SuperSonicAI is to have a working and compatible OS and installing Git. Git can be installed on any modern operating system. Download and installation instructions can be found on their website: <a href="https://www.git-scm.com" className="text-yellow-400 hover:text-yellow-600 visited:text-purple-500 no-underline">https://www.git-scm.com/</a></p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                git --version
            </SyntaxHighlighter>
            <p className="text-center">Verify that git is installed by opening a terminal and running the above command. If git is installed, it will print the installed version of git.</p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Cloning the repo</h2>
            <p className="text-center">Clone the repo to get all the source code and data of the SuperSonicAI. In your terminal, run the command below to move into a directory where you want to put the project.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                cd
            </SyntaxHighlighter>
            <p className="text-center"> From that directory, run the git clone command below to download the repository.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                git clone git@git.cs.vt.edu:dmath010/supersonicai.git
            </SyntaxHighlighter>
            <p className="text-center">Once you have downloaded all of the files, change directory into the project directory of SuperSonicAI by calling the command:</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                cd supersonicai/
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Installing Python</h2>
            <p className="text-center">SuperSonicAI, needs an installation of python versions 3.7, 3.8 or 3.9. Newer and older versions of python are not compatible with all the software dependencies of SuperSonicAI. Installation instructions can be found on the python website <a href="https://www.python.org/" className="text-yellow-400 hover:text-yellow-600 visited:text-purple-500 no-underline"></a>https://www.python.org/.</p>
            <p className="text-center">Verify installation by calling the below. If installed correctily, a version number will be printed.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python --version
            </SyntaxHighlighter>
            <p className="text-center">To use specific versions of python, append the version number like the following:</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python3.8 --version
            </SyntaxHighlighter>
            <p className="text-center">or</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python3.8.exe --version
            </SyntaxHighlighter>
            <p className="text-center">This will run python version 3.8 assuming it is installed. Once installed, the software packages can be installed using pip. The packages needed are listed in requirements.txt found in the directory of the git repo. You can install all of the required packages by running the command below.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                pip install -r requirements.txt
            </SyntaxHighlighter>
            <p>You can install individual packages by using the command below.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                {packages}
            </SyntaxHighlighter>
            <p>Verify the package installations by running the below command and checking that all packages listed under requirements.txt are printed by pip list.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                pip list
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Games ROM</h2>
            <p className="text-center">SuperSonicAI is run from a small set of driver scripts. These scripts perform common tasks like training agents and demonstrating their performance and recording playbacks into mp4 videos. But before running these scripts, the games read-only memory needs to be added to the project. Without this, the emulator will not have the executables or data to run the game engine itself. Users will first need to download Sonic the Hedgehog via STEAM. Then, games can be imported by calling the below command for local ROM files</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python scripts/import_path.py
            </SyntaxHighlighter>
            <p className="text-center">or this command for STEAM ROM files.</p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python scripts/import_sega_classics.py
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Random Agent</h2>
            <p className="text-center">Random Agent was the initial model utilized to test our software environment. It requires no training and simply seeks to ensure that our teams&apos; system capabilities align with the OpenAI GymRetro Interface. Random Agent renders an environment and creates an agent that randomly chooses 1 action per timestep from the predefined 7 moves in our action space. </p>
            <SyntaxHighlighter
                style={vscDarkPlus}
                language="bash">
                python source/models/front_end_random.py
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Random Agent Code</h2>
            <p>source/models/front_end_random.py</p>
            <SyntaxHighlighter
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {randomAgent}
            </SyntaxHighlighter>
          </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/"}>Phase 2: Model Experimentation -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Initialization;
