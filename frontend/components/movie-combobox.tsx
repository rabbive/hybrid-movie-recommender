"use client";

import * as React from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface MovieComboboxProps {
  movies: string[];
  value: string;
  onChange: (value: string) => void;
}

export function MovieCombobox({ movies, value, onChange }: MovieComboboxProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between bg-zinc-800 border-zinc-700 text-zinc-100 hover:bg-zinc-700 hover:text-zinc-100 h-10 font-normal"
        >
          <span className="truncate">
            {value || "Select a seed movie…"}
          </span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-40" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[420px] p-0 bg-zinc-900 border-zinc-700">
        <Command className="bg-zinc-900">
          <CommandInput
            placeholder="Search movies…"
            className="text-zinc-100 border-zinc-700"
          />
          <CommandList>
            <CommandEmpty className="text-zinc-500 text-sm">
              No movie found.
            </CommandEmpty>
            <CommandGroup>
              {movies.map((movie) => (
                <CommandItem
                  key={movie}
                  value={movie}
                  onSelect={(currentValue) => {
                    onChange(currentValue === value ? "" : currentValue);
                    setOpen(false);
                  }}
                  className="text-zinc-300 data-[selected=true]:bg-zinc-800 data-[selected=true]:text-zinc-100 cursor-pointer"
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      value === movie
                        ? "opacity-100 text-emerald-400"
                        : "opacity-0"
                    )}
                  />
                  {movie}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
