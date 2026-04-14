"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const API = "http://localhost:8000";

interface Recommendation {
  title: string;
  movieId: number;
  similarity: number;
  predicted_rating: number | null;
  pred_norm: number | null;
  sim_norm: number | null;
  final_score: number;
}

interface HistBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

function titleCase(str: string) {
  return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

const chartTheme = {
  bg: "#18181b",
  grid: "#27272a",
  text: "#a1a1aa",
  tooltip: { bg: "#09090b", border: "#3f3f46", text: "#e4e4e7" },
};

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-md border px-3 py-2 text-xs"
      style={{
        background: chartTheme.tooltip.bg,
        borderColor: chartTheme.tooltip.border,
        color: chartTheme.tooltip.text,
      }}
    >
      <p className="font-medium">{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey}>
          {p.name ?? p.dataKey}: {typeof p.value === "number" ? p.value.toLocaleString() : p.value}
        </p>
      ))}
    </div>
  );
}

/* ---------- 1. Rating Distribution ---------- */
export function RatingsDistributionChart() {
  const [data, setData] = useState<{ rating: number; count: number }[] | null>(null);

  useEffect(() => {
    fetch(`${API}/charts/ratings-distribution`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) return null;

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">Distribution of Ratings</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis dataKey="rating" tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <YAxis tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill="#4682b4" radius={[2, 2, 0, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ---------- 2. Ratings Per User ---------- */
export function RatingsPerUserChart() {
  const [data, setData] = useState<HistBin[] | null>(null);

  useEffect(() => {
    fetch(`${API}/charts/ratings-per-user`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) return null;

  const formatted = data.map((d) => ({
    label: `${d.bin_start}–${d.bin_end}`,
    count: d.count,
  }));

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">Ratings Per User</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={formatted}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis dataKey="label" tick={{ fill: chartTheme.text, fontSize: 9 }} interval="preserveStartEnd" />
            <YAxis tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill="#ff7f50" radius={[2, 2, 0, 0]} name="Users" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ---------- 3. Cosine Similarities for Seed Movie ---------- */
export function CosineSimilaritiesChart({ movieTitle }: { movieTitle: string }) {
  const [data, setData] = useState<HistBin[] | null>(null);

  useEffect(() => {
    if (!movieTitle) return;
    fetch(`${API}/charts/cosine-similarities?movie_title=${encodeURIComponent(movieTitle)}`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, [movieTitle]);

  if (!data || !movieTitle) return null;

  const formatted = data.map((d) => ({
    label: d.bin_start.toFixed(2),
    count: d.count,
  }));

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">
          Cosine Similarity Distribution — {titleCase(movieTitle)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={formatted}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis dataKey="label" tick={{ fill: chartTheme.text, fontSize: 9 }} interval="preserveStartEnd" />
            <YAxis tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" fill="#2e8b57" radius={[2, 2, 0, 0]} name="Movies" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ---------- 4. Similarity (Top Picks) ---------- */
export function SimilarityChart({ recommendations }: { recommendations: Recommendation[] }) {
  const data = [...recommendations]
    .reverse()
    .map((r) => ({ title: titleCase(r.title), similarity: parseFloat(r.similarity.toFixed(3)) }));

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">Content: Cosine Similarity</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis type="number" tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <YAxis dataKey="title" type="category" tick={{ fill: chartTheme.text, fontSize: 10 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="similarity" fill="#008080" radius={[0, 2, 2, 0]} name="Similarity" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ---------- 5. Hybrid Final Score (Top Picks) ---------- */
export function FinalScoreChart({ recommendations }: { recommendations: Recommendation[] }) {
  const data = [...recommendations]
    .reverse()
    .map((r) => ({ title: titleCase(r.title), final_score: parseFloat(r.final_score.toFixed(3)) }));

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">Hybrid Score (After Normalization)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis type="number" tick={{ fill: chartTheme.text, fontSize: 11 }} />
            <YAxis dataKey="title" type="category" tick={{ fill: chartTheme.text, fontSize: 10 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="final_score" fill="#6a5acd" radius={[0, 2, 2, 0]} name="Final Score" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ---------- 6. Predicted Rating (Top Picks) ---------- */
export function PredictedRatingChart({ recommendations }: { recommendations: Recommendation[] }) {
  const withRatings = recommendations.filter((r) => r.predicted_rating !== null);
  if (withRatings.length === 0) return null;

  const data = [...withRatings]
    .reverse()
    .map((r) => ({ title: titleCase(r.title), predicted_rating: parseFloat(r.predicted_rating!.toFixed(2)) }));

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300">Collaborative: Predicted Rating (SVD)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
            <XAxis type="number" tick={{ fill: chartTheme.text, fontSize: 11 }} domain={[0, 5]} />
            <YAxis dataKey="title" type="category" tick={{ fill: chartTheme.text, fontSize: 10 }} width={120} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="predicted_rating" fill="#ffa500" radius={[0, 2, 2, 0]} name="Predicted Rating" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
