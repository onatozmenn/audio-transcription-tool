"use client";

import * as Select from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { forwardRef } from "react";

export type LanguageSelectOption = {
  countryCode: string;
  label: string;
  value: string;
};

type LanguageSelectProps = {
  id: string;
  onValueChange: (value: string) => void;
  options: LanguageSelectOption[];
  placeholder: string;
  value: string | null;
};

function FlagIcon({ countryCode }: { countryCode: string }) {
  return (
    <span
      aria-hidden="true"
      className={`fi fi-${countryCode} inline-block shrink-0 rounded-[2px] shadow-[0_0_0_1px_rgba(255,255,255,0.12)]`}
    />
  );
}

export const LanguageSelect = forwardRef<HTMLButtonElement, LanguageSelectProps>(
  function LanguageSelect(
    { id, onValueChange, options, placeholder, value },
    forwardedRef,
  ) {
    const selectedOption = options.find((option) => option.value === value) ?? null;

    return (
      <Select.Root value={value ?? ""} onValueChange={onValueChange}>
        <Select.Trigger
          id={id}
          ref={forwardedRef}
          className={[
            "flex h-10 w-full items-center justify-between gap-3 rounded-lg border py-2 pl-3 pr-3 text-left text-sm font-semibold outline-none transition-colors",
            "focus-visible:ring-2 focus-visible:ring-emerald-200/45 data-[state=open]:border-emerald-200/45 data-[state=open]:ring-2 data-[state=open]:ring-emerald-200/20",
            selectedOption
              ? "border-neutral-600/80 bg-[#171a18] text-neutral-100 hover:border-neutral-500"
              : "border-neutral-700/80 bg-[#151816] text-neutral-400 hover:border-neutral-600 hover:text-neutral-200",
          ].join(" ")}
        >
          <Select.Value placeholder={placeholder}>
            {selectedOption ? (
              <span className="flex min-w-0 items-center gap-2.5">
                <FlagIcon countryCode={selectedOption.countryCode} />
                <span className="truncate">{selectedOption.label}</span>
              </span>
            ) : null}
          </Select.Value>
          <Select.Icon asChild>
            <ChevronDown aria-hidden="true" className="size-3.5 shrink-0 text-neutral-400" />
          </Select.Icon>
        </Select.Trigger>

        <Select.Portal>
          <Select.Content
            position="popper"
            sideOffset={6}
            collisionPadding={12}
            className="z-[100] max-h-[min(24rem,var(--radix-select-content-available-height))] min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-white/[0.12] bg-[#151816] shadow-[0_12px_28px_rgba(0,0,0,0.38)]"
          >
            <Select.ScrollUpButton className="flex h-7 items-center justify-center text-neutral-400">
              <ChevronUp aria-hidden="true" className="size-4" />
            </Select.ScrollUpButton>

            <Select.Viewport className="p-1">
              {options.map((option) => (
                <Select.Item
                  key={option.value}
                  value={option.value}
                  className="relative flex h-9 cursor-default select-none items-center gap-2.5 rounded-md py-2 pl-3 pr-8 text-sm text-neutral-300 outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-white/[0.07] data-[highlighted]:text-white data-[state=checked]:bg-emerald-300/[0.08] data-[state=checked]:text-emerald-100"
                >
                  <FlagIcon countryCode={option.countryCode} />
                  <Select.ItemText>{option.label}</Select.ItemText>
                  <Select.ItemIndicator className="absolute right-2.5 inline-flex items-center text-emerald-300">
                    <Check aria-hidden="true" className="size-3.5" />
                  </Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.Viewport>

            <Select.ScrollDownButton className="flex h-7 items-center justify-center text-neutral-400">
              <ChevronDown aria-hidden="true" className="size-4" />
            </Select.ScrollDownButton>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    );
  },
);
