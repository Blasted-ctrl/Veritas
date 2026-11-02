import Image from "next/image";
import Link from "next/link";

export function Brand({ small = false }: { small?: boolean }) {
  return (
    <Link
      className={"wordmark" + (small ? " small" : "")}
      href="/"
      aria-label="Veritas home"
    >
      <Image
        className="brand-mark"
        src="/icon.svg"
        width={44}
        height={44}
        alt=""
        aria-hidden="true"
      />
      <span>Veritas</span>
    </Link>
  );
}
