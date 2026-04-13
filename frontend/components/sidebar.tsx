import {
  LayoutDashboard,
  Settings,
  Film,
  BarChart3,
  Info,
  Clapperboard,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { icon: LayoutDashboard, label: "Overview", active: true },
  { icon: Settings, label: "Settings", active: false },
  { icon: Film, label: "Movies", active: false },
  { icon: BarChart3, label: "Analytics", active: false },
  { icon: Info, label: "About", active: false },
];

export function Sidebar() {
  return (
    <div className="w-56 border-r border-zinc-800 bg-zinc-900/50 flex flex-col shrink-0">
      {/* Brand */}
      <div className="p-4 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-emerald-500 flex items-center justify-center shrink-0">
            <Clapperboard className="w-4 h-4 text-zinc-950" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-zinc-100 truncate">
              Movie Rec
            </p>
            <p className="text-xs text-zinc-500">Hybrid System</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-2 space-y-0.5">
        {navItems.map((item) => (
          <div
            key={item.label}
            className={cn(
              "flex items-center gap-2.5 px-3 py-2 rounded-md text-sm cursor-default transition-colors select-none",
              item.active
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            )}
          >
            <item.icon className="w-4 h-4 shrink-0" />
            {item.label}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-zinc-800">
        <p className="text-xs text-zinc-600">TF-IDF · SVD · Streamlit</p>
        <p className="text-xs text-zinc-700 mt-0.5">Academic project</p>
      </div>
    </div>
  );
}
