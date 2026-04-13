import { Card, CardContent } from "@/components/ui/card";
import { Users, Film, Star, Activity } from "lucide-react";

interface StatsData {
  dataset_stats: Record<string, number> | null;
  cv_metrics: {
    test_rmse_mean: number;
    test_rmse_std: number;
    n_splits: number;
  } | null;
  rmse: number | null;
}

export function StatsCards({ stats }: { stats: StatsData }) {
  const ds = stats.dataset_stats;

  const items = [
    {
      icon: Users,
      label: "Users",
      value: ds?.n_users?.toLocaleString() ?? "—",
      sub: "in ratings dataset",
      iconColor: "text-blue-400",
      iconBg: "bg-blue-950/40",
    },
    {
      icon: Film,
      label: "Movies",
      value: ds?.n_movies_after_join?.toLocaleString() ?? "—",
      sub: "after TMDB join",
      iconColor: "text-violet-400",
      iconBg: "bg-violet-950/40",
    },
    {
      icon: Star,
      label: "Ratings",
      value: ds?.n_ratings?.toLocaleString() ?? "—",
      sub: ds
        ? `${((1 - ds.sparsity) * 100).toFixed(1)}% matrix density`
        : "—",
      iconColor: "text-amber-400",
      iconBg: "bg-amber-950/40",
    },
    {
      icon: Activity,
      label: "Test RMSE",
      value: stats.rmse !== null ? stats.rmse.toFixed(4) : "—",
      sub: stats.cv_metrics
        ? `CV ${stats.cv_metrics.test_rmse_mean.toFixed(3)} ± ${stats.cv_metrics.test_rmse_std.toFixed(3)}`
        : "single-split holdout",
      iconColor: "text-emerald-400",
      iconBg: "bg-emerald-950/40",
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {items.map((item) => (
        <Card key={item.label} className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-2">
              <div className={`p-2 rounded-md ${item.iconBg} shrink-0`}>
                <item.icon className={`w-4 h-4 ${item.iconColor}`} />
              </div>
              <p className="text-xs text-zinc-500 font-medium">{item.label}</p>
            </div>
            <p className="text-2xl font-semibold text-zinc-100 tabular-nums">
              {item.value}
            </p>
            <p className="text-xs text-zinc-600 mt-1">{item.sub}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
