"use client";

import { useState, useEffect } from "react";
import { StatsCards } from "@/components/stats-cards";
import { MovieCombobox } from "@/components/movie-combobox";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Film,
  Shuffle,
  Star,
  TrendingUp,
  Layers,
  Database,
  Sparkles,
  Users,
  AlertCircle,
  Loader2,
  BarChart3,
} from "lucide-react";
import {
  RatingsDistributionChart,
  RatingsPerUserChart,
  CosineSimilaritiesChart,
  SimilarityChart,
  FinalScoreChart,
  PredictedRatingChart,
} from "@/components/analysis-charts";

const API = "http://localhost:8000";

interface StatsData {
  dataset_stats: Record<string, number> | null;
  cv_metrics: {
    test_rmse_mean: number;
    test_rmse_std: number;
    n_splits: number;
  } | null;
  rmse: number | null;
}

interface Recommendation {
  title: string;
  movieId: number;
  similarity: number;
  predicted_rating: number | null;
  pred_norm: number | null;
  sim_norm: number | null;
  final_score: number;
}

function titleCase(str: string) {
  return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Home() {
  const [movies, setMovies] = useState<string[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [selectedMovie, setSelectedMovie] = useState("");
  const [userId, setUserId] = useState(1);
  const [recommendations, setRecommendations] = useState<
    Recommendation[] | null
  >(null);
  const [seedLabel, setSeedLabel] = useState("");
  const [loading, setLoading] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/movies`).then((r) => r.json()),
      fetch(`${API}/stats`).then((r) => r.json()),
    ])
      .then(([moviesData, statsData]) => {
        setMovies(moviesData.movies);
        setStats(statsData);
        if (moviesData.movies.length > 0) setSelectedMovie(moviesData.movies[0]);
      })
      .catch(() =>
        setError(
          "Cannot reach the API. Run: uvicorn api:app --reload --port 8000"
        )
      )
      .finally(() => setBootstrapping(false));
  }, []);

  async function runRecommend(movieTitle: string) {
    if (!movieTitle) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API}/recommend?user_id=${userId}&movie_title=${encodeURIComponent(movieTitle)}&top_n=5`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? res.statusText);
      }
      setRecommendations(await res.json());
      setSeedLabel(movieTitle);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function handleSurprise() {
    if (!movies.length) return;
    const pick = movies[Math.floor(Math.random() * movies.length)];
    setSelectedMovie(pick);
    runRecommend(pick);
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">Overview</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Hybrid movie recommendations — TF-IDF content filtering + SVD
          collaborative filtering
        </p>
      </div>

      <Separator className="bg-zinc-800" />

      {/* Algorithm cards — mirrors "plan" cards in reference UI */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base text-zinc-100">
                Content-Based
              </CardTitle>
              <Badge className="bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-800">
                weight: 40%
              </Badge>
            </div>
            <CardDescription className="text-zinc-500">
              TF-IDF vectorization on genres, keywords, cast, director &amp;
              overview. Ranks candidates by cosine similarity.
            </CardDescription>
          </CardHeader>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base text-zinc-100">
                Collaborative
              </CardTitle>
              <Badge className="bg-emerald-950 text-emerald-400 border-emerald-900 hover:bg-emerald-950">
                weight: 60%
              </Badge>
            </div>
            <CardDescription className="text-zinc-500">
              SVD matrix factorization on 100k MovieLens ratings. Predicts
              personalized scores for warm users.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>

      {/* Get recommendations — mirrors "On-Demand Usage" section */}
      <Card className="bg-zinc-900 border-zinc-800 border-dashed">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <CardTitle className="text-base text-zinc-100">
              Get Recommendations
            </CardTitle>
          </div>
          <CardDescription className="text-zinc-500">
            Pick a seed movie and user ID — warm users get personalised hybrid
            scores, new users get content-only results.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {bootstrapping ? (
            <div className="flex items-center gap-2 text-sm text-zinc-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading model data…
            </div>
          ) : (
            <>
              <div className="flex gap-3">
                <div className="flex-1">
                  <MovieCombobox
                    movies={movies}
                    value={selectedMovie}
                    onChange={setSelectedMovie}
                  />
                </div>
                <div className="w-36 relative">
                  <Users className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
                  <Input
                    type="number"
                    value={userId}
                    onChange={(e) => setUserId(Number(e.target.value))}
                    min={1}
                    className="pl-9 bg-zinc-800 border-zinc-700 text-zinc-100 h-10"
                    placeholder="User ID"
                  />
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={() => runRecommend(selectedMovie)}
                  disabled={loading || !selectedMovie}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white border-0"
                >
                  {loading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Film className="w-4 h-4" />
                  )}
                  {loading ? "Loading…" : "Recommend"}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleSurprise}
                  disabled={loading || !movies.length}
                  className="border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                >
                  <Shuffle className="w-4 h-4" />
                  Surprise me
                </Button>
              </div>
            </>
          )}

          {error && (
            <div className="flex items-start gap-2 text-sm text-red-400 bg-red-950/30 border border-red-900/50 rounded-lg px-3 py-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dataset stats — mirrors "AI Line Edits" section */}
      {stats && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-zinc-400">
            Dataset &amp; Model
          </h2>
          <StatsCards stats={stats} />
        </div>
      )}

      {/* Data sources — mirrors "Source Control" section */}
      <div className="space-y-2">
        <h2 className="text-sm font-medium text-zinc-400">Data Sources</h2>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-0">
            {[
              {
                name: "MovieLens",
                desc: "ratings_small.csv · links_small.csv",
                color: "bg-blue-500",
              },
              {
                name: "TMDB 5000",
                desc: "tmdb_5000_movies.csv · tmdb_5000_credits.csv",
                color: "bg-teal-500",
              },
            ].map((src, i, arr) => (
              <div
                key={src.name}
                className={`flex items-center gap-3 px-4 py-3 ${
                  i < arr.length - 1 ? "border-b border-zinc-800" : ""
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-md ${src.color}/20 flex items-center justify-center shrink-0`}
                >
                  <Database className={`w-4 h-4 text-${src.color.replace("bg-", "")}`} />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-zinc-200">
                    {src.name}
                  </p>
                  <p className="text-xs text-zinc-500 truncate">{src.desc}</p>
                </div>
                <Badge
                  variant="outline"
                  className="ml-auto border-zinc-700 text-zinc-500 text-xs"
                >
                  Local
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Recommendation results */}
      {recommendations && recommendations.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-zinc-400">
              Results for
            </h2>
            <Badge className="bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-800">
              {titleCase(seedLabel)}
            </Badge>
            <span className="text-xs text-zinc-600">
              · User {userId}
              {recommendations[0].predicted_rating === null
                ? " (cold-start)"
                : " (warm)"}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {recommendations.map((rec, i) => (
              <Card
                key={rec.movieId}
                className="bg-zinc-900 border-zinc-800 hover:border-zinc-600 transition-colors group"
              >
                <CardHeader className="pb-2 pt-4 px-4">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-zinc-600 font-mono">
                      #{i + 1}
                    </span>
                    <Badge className="bg-emerald-950 text-emerald-400 border-emerald-900/50 hover:bg-emerald-950 text-xs tabular-nums">
                      {(rec.final_score * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <CardTitle className="text-sm font-medium text-zinc-100 leading-snug capitalize">
                    {rec.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-4 pb-4 space-y-1.5">
                  <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                    <Layers className="w-3 h-3 shrink-0" />
                    <span>Similarity {rec.similarity.toFixed(3)}</span>
                  </div>
                  {rec.predicted_rating !== null ? (
                    <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                      <Star className="w-3 h-3 shrink-0" />
                      <span>Predicted {rec.predicted_rating.toFixed(2)}</span>
                    </div>
                  ) : (
                    <div className="text-xs text-zinc-600 italic">
                      Content-only
                    </div>
                  )}
                  <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                    <TrendingUp className="w-3 h-3 shrink-0" />
                    <span>Score {rec.final_score.toFixed(3)}</span>
                  </div>

                  {/* Score bar */}
                  <div className="mt-2 h-1 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all"
                      style={{ width: `${rec.final_score * 100}%` }}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Figures for Analysis */}
      <Separator className="bg-zinc-800" />
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-zinc-400" />
          <h2 className="text-sm font-medium text-zinc-400">
            Figures for Analysis
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <RatingsDistributionChart />
          <RatingsPerUserChart />
        </div>

        {selectedMovie && (
          <CosineSimilaritiesChart movieTitle={selectedMovie} />
        )}

        {recommendations && recommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SimilarityChart recommendations={recommendations} />
            <FinalScoreChart recommendations={recommendations} />
          </div>
        )}

        {recommendations && recommendations.length > 0 && (
          <PredictedRatingChart recommendations={recommendations} />
        )}
      </div>
    </div>
  );
}
