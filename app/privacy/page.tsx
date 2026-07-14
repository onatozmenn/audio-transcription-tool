import type { Metadata } from "next";
import Link from "next/link";
import { ShieldCheck } from "lucide-react";

export const metadata: Metadata = {
    title: "Privacy Policy - Audio Transcription",
    description: "How local and cloud audio transcription data is processed, protected, and deleted.",
};

const Section = ({
    title,
    children,
}: {
    title: string;
    children: React.ReactNode;
}) => (
    <div className="space-y-3">
        <h2 className="font-display text-base font-semibold text-neutral-100">{title}</h2>
        <div className="space-y-2 text-sm leading-7 text-neutral-400">{children}</div>
    </div>
);

export default function PrivacyPolicy() {
    const updated = "July 14, 2026";

    return (
        <main className="flex min-h-screen items-start justify-center px-4 py-12 sm:px-6">
            <article className="w-full max-w-2xl rounded-lg border border-white/[0.09] bg-[#171a18]/95 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.35)] sm:p-10">

                {/* Back link */}
                <Link
                    href="/"
                    className="mb-8 inline-flex items-center gap-1.5 text-xs text-neutral-500 transition-colors hover:text-neutral-300"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" height="14px" viewBox="0 -960 960 960" width="14px" fill="currentColor">
                        <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
                    </svg>
                    Back to app
                </Link>

                {/* Header */}
                <header className="mb-8 space-y-3">
                    <p className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
                        <ShieldCheck className="size-3.5" />
                        Privacy-first by design
                    </p>
                    <h1 className="font-display text-2xl font-semibold text-white">
                        Privacy Policy
                    </h1>
                    <p className="text-xs text-neutral-500">
                        Last updated: {updated}
                    </p>
                </header>

                {/* Sections */}
                <div className="divide-y divide-white/5 space-y-8">

                    <div className="pt-8 space-y-3">
                        <Section title="Overview">
                            <p>
                                This tool does not require an account and does not use advertising or analytics trackers.
                                Limited data is processed only to provide and protect the transcription service. How your audio is handled depends on the device you use:
                            </p>
                            <ul className="list-disc space-y-2 pl-5">
                                <li>
                                    <span className="font-medium text-neutral-200">On Desktop:</span> All transcription happens entirely inside your browser using Whisper Small or Moonshine AI models. No audio, text, or personal information is ever sent to a server.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">On Mobile:</span> Due to hardware limitations, your audio is uploaded to temporary Vercel Blob storage, transcribed by <a href="https://groq.com" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300">Groq Cloud</a> (Whisper Large V3), and deleted after processing.
                                </li>
                            </ul>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="How Transcription Works">
                            <p>
                                <span className="font-medium text-neutral-200">Desktop:</span> Audio is decoded and processed entirely within your browser. English audio uses the Moonshine Base model (fast, lightweight); other languages use the Whisper Small model. Both run via WebGPU (with WASM fallback). The AI model is downloaded once from Hugging Face and cached in your browser — after that, transcription works fully offline.
                            </p>
                            <p className="mt-2">
                                <span className="font-medium text-neutral-200">Mobile:</span> Audio is chunked and uploaded directly from your device into a temporary Vercel Blob object. The server gives Groq time-limited access for transcription. Successful jobs are deleted immediately. Failed or abandoned uploads are removed by a scheduled cleanup after the four-hour session lifetime plus a one-hour grace period.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Abuse Prevention & Request Metadata">
                            <p>
                                Mobile API requests are checked by Vercel BotID and rate limited. The application converts the request IP address into a keyed HMAC value before placing it in the shared rate-limit store; the raw IP address is not stored by the application. Rate-limit entries expire after at most 30 minutes.
                            </p>
                            <p>
                                Hosting and API providers may process standard network logs under their own privacy and retention policies. A separate random secret is used for transcription grants and no rate-limit identifier is included in transcript output.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Export & Continue with AI">
                            <p>
                                You can export your transcript as plain text, timestamped text, Word (.docx), or PDF. All document generation happens entirely in your browser — no data is sent to any server for export.
                            </p>
                            <p className="mt-2">
                                The &quot;Continue with AI&quot; buttons copy your transcript to clipboard and open a third-party AI chat (ChatGPT, Claude, Gemini, or Grok) in a new tab. We do not send your transcript to these services — you paste it yourself. Each service&apos;s own privacy policy applies once you interact with them.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Browser Cache & Local Storage">
                            <p>
                                The AI model weights (approximately 50–150 MB depending on language) are cached in your browser&apos;s native cache after the first download. You can clear this cache at any time through your browser settings.
                            </p>
                            <p>
                                We use minimal{" "}
                                <code className="rounded bg-neutral-800 px-1 py-0.5 text-xs font-mono text-neutral-300">localStorage</code>{" "}
                                flags to remember whether the model was previously loaded (to skip redundant loading UI). We do not store audio, transcripts, personal data, or tracking identifiers in local storage.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Third-Party Services">
                            <ul className="list-disc space-y-2 pl-5">
                                <li>
                                    <span className="font-medium text-neutral-200">Hugging Face (Desktop):</span> Used only to download open-source AI model files on first use. Their <a href="https://huggingface.co/privacy" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">privacy policy</a> applies.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">Vercel Blob and Vercel BotID (Mobile):</span> Temporary storage and abuse prevention for mobile requests. Uploads are deleted after processing or by scheduled stale-file cleanup.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">Upstash Redis (Mobile):</span> Stores only keyed, pseudonymous rate-limit counters with a maximum 30-minute lifetime. Audio and transcripts are never stored there.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">Groq Cloud (Mobile):</span> Processes mobile audio using Whisper Large V3. Groq&apos;s terms and retention policy apply while it processes the request. See the <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">Groq Privacy Policy</a>.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">AI Chat Services (Optional):</span> ChatGPT, Claude, Gemini, and Grok are only opened if you click a &quot;Continue with AI&quot; button. We do not send data to them — you paste it yourself.
                                </li>
                            </ul>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Children's Privacy">
                            <p>
                                This service is not directed at children and does not knowingly create profiles about children. The same limited processing described above applies to every request.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Changes to This Policy">
                            <p>
                                If this policy changes, the updated version will be published at this URL with a revised &quot;Last updated&quot; date.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Contact">
                            <p>
                                Questions? Reach out via{" "}
                                <a
                                    href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300"
                                >
                                    LinkedIn
                                </a>
                                .
                            </p>
                        </Section>
                    </div>

                </div>

                {/* Footer */}
                <div className="mt-10 flex items-center justify-between border-t border-white/10 pt-6">
                    <p className="text-xs text-neutral-500">
                        Developed by{" "}
                        <span className="font-medium text-neutral-300">Onat Özmen</span>
                    </p>
                    <Link
                        href="/"
                        className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 hover:text-cyan-200"
                    >
                        Back to app
                    </Link>
                </div>

            </article>
        </main>
    );
}
