interface Props {
  label: string;
  value: number; // 0..1
  tone: "real" | "fake" | "neutral";
}

const TONE = {
  real: "bg-real",
  fake: "bg-fake",
  neutral: "bg-sky-400",
} as const;

export function ConfidenceBar({ label, value, tone }: Props) {
  const pct = Math.round(value * 100);
  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-sm text-slate-300">
        <span>{label}</span>
        <span className="font-mono tabular-nums">{pct}%</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full transition-[width] duration-500 ease-out ${TONE[tone]}`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}
