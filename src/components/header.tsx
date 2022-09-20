import { NextComponentType } from "next";
import Link from "next/link";

const Header: NextComponentType = () => {

    return (
        <article>
            <header className="header bg-gray-900 flex w-full p-8 flex-col items-center gap-2 md:justify-between md:flex-row ">
                <Link href={"/"}>
                    <h1 className="text-4xl font-bold cursor-pointer text-purple-500">SuperSonicAI</h1>
                </Link>
                <div className="gap-2 flex flex-row">
                    <Link href={"/about"}>
                        <button className=" rounded bg-slate-700 p-2">Our Story</button>
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