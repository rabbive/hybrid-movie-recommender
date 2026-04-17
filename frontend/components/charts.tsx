"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface HistogramData {
  counts: number[];
  bin_starts: number[];
  bin_ends: number[];
}

interface HorizontalBarDatum {
  label: string;
  value: number;
}

function formatTick(value: number) {
  if (Math.abs(value) >= 1000) return value.toLocaleString();
  return Number.isInteger(value) ? `${value}` : value.toFixed(2);
}

export function HistogramCard({
  title,
  subtitle,
  data,
  colorClass,
}: {
  title: string;
  subtitle: string;
  data: HistogramData | null;
  colorClass: string;
}) {
  const max = Math.max(...(data?.counts ?? [0]), 1);

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-100">{title}</CardTitle>
        <p className="text-xs text-zinc-500">{subtitle}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        {!data ? (
          <p className="text-sm text-zinc-500">Chart data unavailable.</p>
        ) : (
          <>
            <div className="flex items-end gap-1 h-44 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
              {data.counts.map((count, index) => (
                <div
                  key={`${data.bin_starts[index]}-${data.bin_ends[index]}`}
                  className="flex-1 flex items-end h-full"
                  title={`${formatTick(data.bin_starts[index])} to ${formatTick(data.bin_ends[index])}: ${count}`}
                >
                  <div
                    className={`w-full rounded-t-sm ${colorClass}`}
                    style={{ height: `${Math.max((count / max) * 100, count > 0 ? 3 : 0)}%` }}
                  />
                </div>
              ))}
            </div>
            <div className="flex justify-between text-[11px] text-zinc-600">
              <span>{formatTick(data.bin_starts[0])}</span>
              <span>{formatTick(data.bin_ends[data.bin_ends.length - 1])}</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function HorizontalBarCard({
  title,
  subtitle,
  data,
  colorClass,
  formatter = (value: number) => value.toFixed(3),
}: {
  title: string;
  subtitle: string;
  data: HorizontalBarDatum[];
  colorClass: string;
  formatter?: (value: number) => string;
}) {
  const max = Math.max(...data.map((item) => item.value), 1);

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-100">{title}</CardTitle>
        <p className="text-xs text-zinc-500">{subtitle}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        {data.length === 0 ? (
          <p className="text-sm text-zinc-500">Run a recommendation to see this chart.</p>
        ) : (
          data.map((item) => (
            <div key={item.label} className="space-y-1.5">
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-zinc-300 truncate">{item.label}</span>
                <span className="text-zinc-500 tabular-nums">{formatter(item.value)}</span>
              </div>
              <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className={`h-full rounded-full ${colorClass}`}
                  style={{ width: `${Math.max((item.value / max) * 100, item.value > 0 ? 4 : 0)}%` }}
                />
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
