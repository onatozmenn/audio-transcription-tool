import type { Metadata } from "next";
import {
  PrivacyPolicyPage,
  type PrivacyFact,
  type PrivacySection,
} from "@/components/privacy-policy-page";

export const metadata: Metadata = {
  title: "Privacy Policy - Audio Transcription",
  description: "How local and cloud audio transcription data is processed, protected, and deleted.",
  alternates: {
    canonical: "https://audio-transcription.app/privacy",
    languages: {
      en: "https://audio-transcription.app/privacy",
      tr: "https://audio-transcription.app/tr/privacy",
    },
  },
};

const facts: PrivacyFact[] = [
  {
    label: "Desktop",
    steps: ["Device", "Browser model", "Transcript"],
    note: "Audio and transcript text never leave your browser.",
  },
  {
    label: "Mobile",
    steps: ["Device", "Temporary Blob", "Groq", "Deletion"],
    note: "Successful files are deleted immediately; abandoned files are removed by scheduled cleanup.",
  },
  {
    label: "Service protection",
    steps: ["BotID", "Keyed HMAC counter", "Expiry"],
    note: "The application does not store raw IP addresses. Rate-limit counters live for at most 30 minutes.",
  },
];

const sections: PrivacySection[] = [
  {
    id: "overview",
    title: "Overview",
    body: (
      <>
        <p>
          This tool does not require an account and does not use advertising or analytics trackers. Limited data is processed only to provide and protect the transcription service. Audio handling depends on the device you use.
        </p>
        <ul>
          <li>
            <strong>On desktop:</strong> Transcription happens entirely inside your browser using Whisper Small or Moonshine AI models. No audio, text, or personal information is sent to a server.
          </li>
          <li>
            <strong>On mobile:</strong> Audio is uploaded to temporary Vercel Blob storage, transcribed by <a href="https://groq.com" target="_blank" rel="noopener noreferrer">Groq Cloud</a> using Whisper Large V3, and deleted after processing.
          </li>
        </ul>
      </>
    ),
  },
  {
    id: "transcription",
    title: "How Transcription Works",
    body: (
      <>
        <p>
          <strong>Desktop:</strong> Audio is decoded and processed entirely within your browser. English audio uses the Moonshine Base model; other languages use the Whisper Small model. Both run through WebGPU with a WASM fallback. The model is downloaded once from Hugging Face and cached in your browser. After that, transcription can work offline.
        </p>
        <p>
          <strong>Mobile:</strong> Audio is chunked and uploaded directly from your device into a temporary Vercel Blob object. The server gives Groq time-limited access for transcription. Successful jobs are deleted immediately. Failed or abandoned uploads are removed by scheduled cleanup after the four-hour session lifetime plus a one-hour grace period.
        </p>
      </>
    ),
  },
  {
    id: "request-metadata",
    title: "Abuse Prevention and Request Metadata",
    body: (
      <>
        <p>
          Mobile API requests are checked by Vercel BotID and rate limited. The application converts the request IP address into a keyed HMAC value before placing it in the shared rate-limit store; the raw IP address is not stored by the application. Rate-limit entries expire after at most 30 minutes.
        </p>
        <p>
          Hosting and API providers may process standard network logs under their own privacy and retention policies. A separate random secret is used for transcription grants, and no rate-limit identifier is included in transcript output.
        </p>
      </>
    ),
  },
  {
    id: "export",
    title: "Export and Continue with AI",
    body: (
      <>
        <p>
          You can export your transcript as plain text, timestamped text, JSON, Word, or PDF. All document generation happens entirely in your browser; no data is sent to a server for export.
        </p>
        <p>
          The &quot;Continue with AI&quot; buttons copy your transcript to the clipboard and open ChatGPT, Claude, Gemini, or Grok in a new tab. We do not send your transcript to these services. You choose whether to paste it, and each service&apos;s own privacy policy applies once you interact with it.
        </p>
      </>
    ),
  },
  {
    id: "browser-storage",
    title: "Browser Cache and Local Storage",
    body: (
      <>
        <p>
          AI model weights, approximately 50-150 MB depending on language, are cached in your browser&apos;s native cache after the first download. You can clear this cache at any time through your browser settings.
        </p>
        <p>
          We use minimal <code>localStorage</code> flags to remember whether the model was previously loaded and skip redundant loading UI. We do not store audio, transcripts, personal data, or tracking identifiers in local storage.
        </p>
      </>
    ),
  },
  {
    id: "third-parties",
    title: "Third-Party Services",
    body: (
      <ul>
        <li>
          <strong>Hugging Face on desktop:</strong> Used only to download open-source AI model files on first use. The <a href="https://huggingface.co/privacy" target="_blank" rel="noopener noreferrer">Hugging Face Privacy Policy</a> applies.
        </li>
        <li>
          <strong>Vercel Blob and Vercel BotID on mobile:</strong> Provide temporary storage and abuse prevention. Uploads are deleted after processing or by scheduled stale-file cleanup.
        </li>
        <li>
          <strong>Upstash Redis on mobile:</strong> Stores only keyed, pseudonymous rate-limit counters with a maximum 30-minute lifetime. Audio and transcripts are never stored there.
        </li>
        <li>
          <strong>Groq Cloud on mobile:</strong> Processes audio using Whisper Large V3. Groq&apos;s terms and retention policy apply while it processes the request. See the <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer">Groq Privacy Policy</a>.
        </li>
        <li>
          <strong>AI chat services when requested:</strong> ChatGPT, Claude, Gemini, and Grok are opened only after you choose a &quot;Continue with AI&quot; action. We do not send transcript data to them automatically.
        </li>
      </ul>
    ),
  },
  {
    id: "children",
    title: "Children&apos;s Privacy",
    body: (
      <p>
        This service is not directed at children and does not knowingly create profiles about children. The same limited processing described above applies to every request.
      </p>
    ),
  },
  {
    id: "changes",
    title: "Changes to This Policy",
    body: (
      <p>
        If this policy changes, the updated version will be published at this URL with a revised &quot;Last updated&quot; date.
      </p>
    ),
  },
  {
    id: "contact",
    title: "Contact",
    body: (
      <p>
        Questions about this policy can be sent through <a href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250" target="_blank" rel="noopener noreferrer">LinkedIn</a>.
      </p>
    ),
  },
];

export default function PrivacyPolicy() {
  return (
    <PrivacyPolicyPage
      activeLanguage="EN"
      alternateHref="/tr/privacy"
      alternateLanguage="TR"
      backHref="/"
      backLabel="Back to app"
      contactLabel="LinkedIn"
      contentsLabel="On this page"
      dataPathTitle="Where your data goes"
      developedBy="Developed by"
      facts={facts}
      languageLabel="Policy language"
      lastUpdatedLabel="Last updated"
      scopeLabel="Scope"
      scopeText="Local on desktop, temporary cloud processing on mobile"
      sections={sections}
      summary="A direct account of what stays on your device, what leaves it during mobile processing, and when temporary files disappear."
      title="Privacy Policy"
      updated="July 14, 2026"
    />
  );
}