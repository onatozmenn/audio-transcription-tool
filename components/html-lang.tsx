"use client";

import { useEffect } from "react";

/** Sets <html lang> for a subtree and restores English when it unmounts. */
export default function HtmlLang({ lang }: { lang: string }) {
  useEffect(() => {
    const root = document.documentElement;
    const previous = root.lang;
    root.lang = lang;
    return () => {
      root.lang = previous && previous !== lang ? previous : "en";
    };
  }, [lang]);

  return null;
}
