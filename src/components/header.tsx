import { NextComponentType } from "next";
import Image from "next/image";
import Link from "next/link";

import sonic from "../../public/sonic.png"

const Header: NextComponentType = () => {

    return (
        <article>
            <header className="header bg-gray-900 flex w-full p-8 flex-col items-center gap-2 md:justify-between md:flex-row ">
                <div className="flex flex-row">
                    <Image alt="Sonic with a thumbs up" src={sonic} width={32} height={43}/>
                    <Link href={"/"}>
                        <h1 className="text-4xl font-bold cursor-pointer text-purple-500">SuperSonicAI</h1>
                    </Link>
                </div>
                
                <div className="gap-2 flex flex-row">
                    <Link href={"/about"}>
                        <button className=" rounded bg-slate-700 p-2">About us</button>
                    </Link>
                    <a target="_blank" href="https://git.cs.vt.edu/dmath010/supersonicai" rel="noopener noreferrer">
                        <button className=" rounded bg-slate-700 p-2">View the Source Code</button>
                    </a>
                </div>
            </header>
        </article>
    )
}

export default Header