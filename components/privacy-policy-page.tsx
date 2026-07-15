import type { ReactNode } from "react";
import Link from "next/link";

export type PrivacyFact = {
  label: string;
  note: string;
  steps: string[];
};

export type PrivacySection = {
  body: ReactNode;
  id: string;
  title: string;
};

type PrivacyPolicyPageProps = {
  activeLanguage: string;
  alternateHref: string;
  alternateLanguage: string;
  backHref: string;
  backLabel: string;
  contactLabel: string;
  contentsLabel: string;
  dataPathTitle: string;
  developedBy: string;
  facts: PrivacyFact[];
  languageLabel: string;
  lastUpdatedLabel: string;
  scopeLabel: string;
  scopeText: string;
  sections: PrivacySection[];
  summary: string;
  title: string;
  updated: string;
};

function WaveformToTextMark() {
  return (
    <svg
      aria-hidden="true"
      className="h-11 w-20 text-emerald-200/90"
      fill="none"
      viewBox="0 0 80 44"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect x="1" y="16" width="3" height="12" rx="1.5" fill="currentColor" />
      <rect x="7" y="9" width="3" height="26" rx="1.5" fill="currentColor" />
      <rect x="13" y="3" width="3" height="38" rx="1.5" fill="currentColor" />
      <rect x="19" y="12" width="3" height="20" rx="1.5" fill="currentColor" />
      <rect x="25" y="7" width="3" height="30" rx="1.5" fill="currentColor" />
      <path
        d="M36 11H77M36 22H68M36 33H59"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2.5"
        opacity="0.55"
      />
    </svg>
  );
}

export function PrivacyPolicyPage({
  activeLanguage,
  alternateHref,
  alternateLanguage,
  backHref,
  backLabel,
  contactLabel,
  contentsLabel,
  dataPathTitle,
  developedBy,
  facts,
  languageLabel,
  lastUpdatedLabel,
  scopeLabel,
  scopeText,
  sections,
  summary,
  title,
  updated,
}: PrivacyPolicyPageProps) {
  return (
    <main className="min-h-screen px-4 py-5 sm:px-6 sm:py-7 lg:px-10">
      <article className="mx-auto w-full max-w-[1120px]">
        <nav aria-label={languageLabel} className="flex items-center justify-between gap-6">
          <Link
            href={backHref}
            className="inline-flex items-center gap-2 text-sm font-medium text-neutral-400 transition-colors hover:text-neutral-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-200/50"
          >
            <span aria-hidden="true" className="text-base text-emerald-200/65">←</span>
            {backLabel}
          </Link>
          <div className="flex items-center gap-2.5 text-xs">
            <span aria-current="page" className="font-semibold text-neutral-100">
              {activeLanguage}
            </span>
            <span aria-hidden="true" className="text-neutral-700">/</span>
            <Link
              href={alternateHref}
              hrefLang={alternateLanguage.toLowerCase()}
              className="font-medium text-neutral-400 transition-colors hover:text-neutral-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-200/50"
            >
              {alternateLanguage}
            </Link>
          </div>
        </nav>

        <header className="pb-12 pt-14 sm:pb-16 sm:pt-20 lg:pb-20 lg:pt-24">
          <WaveformToTextMark />
          <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_17rem] lg:items-end lg:gap-14">
            <div>
              <h1 className="max-w-3xl font-display text-4xl font-semibold leading-[1.08] text-white sm:text-5xl">
                {title}
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-neutral-400 sm:text-lg sm:leading-8">
                {summary}
              </p>
            </div>
            <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-sm lg:grid-cols-1 lg:gap-y-1">
              <dt className="text-neutral-400">{lastUpdatedLabel}</dt>
              <dd className="font-medium text-neutral-300 lg:mb-4">{updated}</dd>
              <dt className="text-neutral-400">{scopeLabel}</dt>
              <dd className="font-medium leading-6 text-neutral-300">{scopeText}</dd>
            </dl>
          </div>
        </header>

        <section aria-labelledby="privacy-data-path" className="-mx-4 bg-[#141a16] px-4 py-8 sm:mx-0 sm:px-8 sm:py-9">
          <h2 id="privacy-data-path" className="font-display text-lg font-semibold text-neutral-100">
            {dataPathTitle}
          </h2>
          <dl className="mt-8 grid gap-8 lg:grid-cols-3 lg:gap-10">
            {facts.map((fact) => (
              <div key={fact.label}>
                <dt className="text-sm font-semibold text-emerald-100/80">{fact.label}</dt>
                <dd className="mt-3">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm font-medium text-neutral-200">
                    {fact.steps.map((step, index) => (
                      <span key={`${fact.label}-${step}`} className="contents">
                        {index > 0 ? (
                          <span aria-hidden="true" className="text-emerald-200/30">→</span>
                        ) : null}
                        <span>{step}</span>
                      </span>
                    ))}
                  </div>
                  <p className="mt-2 text-sm leading-6 text-neutral-400">{fact.note}</p>
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <div className="grid gap-12 py-14 sm:py-20 lg:grid-cols-[13rem_minmax(0,1fr)] lg:gap-20 lg:py-24">
          <aside className="hidden lg:block">
            <nav aria-label={contentsLabel} className="sticky top-8">
              <p className="text-sm font-semibold text-neutral-300">{contentsLabel}</p>
              <ol className="mt-5 space-y-3">
                {sections.map((section, index) => (
                  <li key={section.id}>
                    <a
                      href={`#${section.id}`}
                      className="grid grid-cols-[1.75rem_1fr] gap-2 text-sm leading-5 text-neutral-400 transition-colors hover:text-neutral-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-200/50"
                    >
                      <span aria-hidden="true" className="tabular-nums text-emerald-200/35">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <span>{section.title}</span>
                    </a>
                  </li>
                ))}
              </ol>
            </nav>
          </aside>

          <div className="min-w-0 space-y-16 sm:space-y-20">
            {sections.map((section, index) => (
              <section key={section.id} id={section.id} className="scroll-mt-8">
                <h2 className="grid grid-cols-[2rem_minmax(0,1fr)] gap-2 font-display text-xl font-semibold leading-7 text-neutral-100 sm:text-2xl sm:leading-8">
                  <span aria-hidden="true" className="pt-0.5 text-sm font-medium tabular-nums text-emerald-200/45">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span>{section.title}</span>
                </h2>
                <div className="privacy-copy mt-5 sm:pl-10">{section.body}</div>
              </section>
            ))}
          </div>
        </div>

        <footer className="flex flex-wrap items-center gap-x-7 gap-y-3 pb-5 text-xs text-neutral-400 sm:pb-7">
          <p>
            {developedBy} <span className="font-medium text-neutral-300">Onat Özmen</span>
          </p>
          <a
            href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-neutral-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-200/50"
          >
            {contactLabel} <span aria-hidden="true">↗</span>
          </a>
          <Link
            href={backHref}
            className="font-medium text-emerald-100/70 transition-colors hover:text-emerald-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-200/50"
          >
            {backLabel}
          </Link>
        </footer>
      </article>
    </main>
  );
}