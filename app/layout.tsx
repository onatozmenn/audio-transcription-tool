import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const CANONICAL_URL = "https://audio-transcription.app";
const TR_URL = `${CANONICAL_URL}/tr`;
const SOCIAL_IMAGE_PATH = "/opengraph-image";
const APP_NAME = "Free Audio Transcription Tool";
const SEO_TITLE = "Free Audio Transcription Tool - Speech to Text Online (No Sign-up)";
const SEO_DESCRIPTION =
  "Convert audio to text online for free with Whisper & Moonshine AI. Transcribe MP3, WAV, M4A, MP4, OGG and more. Desktop runs locally (Moonshine for English, Whisper for other languages); mobile uses secure cloud transcription.";
const SEO_KEYWORDS = [
  "free audio transcription tool",
  "audio transcription online",
  "audio to text converter",
  "speech to text online",
  "speech to text free",
  "transcribe audio to text",
  "transcribe mp3 to text",
  "transcribe wav to text",
  "transcribe m4a to text",
  "transcribe mp4 audio to text",
  "voice to text converter",
  "voice note to text",
  "voice memo transcription",
  "lecture recording transcription",
  "meeting recording transcription",
  "interview transcription tool",
  "podcast transcription",
  "subtitle transcript generator",
  "whisper ai transcription",
  "whisper speech to text",
  "moonshine ai transcription",
  "moonshine speech to text",
  "browser based transcription",
  "client side transcription",
  "private audio transcription",
  "no sign up transcription",
  "free transcription no account",
  "mobile audio transcription",
  "desktop local transcription",
  "multilingual audio transcription",
  "online transcript generator",
  "audio transcript generator",
];

export const metadata: Metadata = {
  metadataBase: new URL(CANONICAL_URL),
  applicationName: APP_NAME,
  title: {
    default: SEO_TITLE,
    template: "%s | Free Audio Transcription Tool (Speech to Text)",
  },
  description: SEO_DESCRIPTION,
  keywords: SEO_KEYWORDS,
  authors: [{ name: "Onat Ozmen" }],
  creator: "Onat Ozmen",
  publisher: "Onat Ozmen",
  alternates: {
    canonical: CANONICAL_URL,
    languages: {
      en: CANONICAL_URL,
      tr: TR_URL,
      "x-default": CANONICAL_URL,
    },
  },
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: SEO_TITLE,
    description: SEO_DESCRIPTION,
    url: CANONICAL_URL,
    siteName: APP_NAME,
    locale: "en_US",
    alternateLocale: ["tr_TR"],
    type: "website",
    images: [
      {
        url: SOCIAL_IMAGE_PATH,
        width: 1200,
        height: 630,
        alt: "Free Audio Transcription Tool preview",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: SEO_TITLE,
    description: SEO_DESCRIPTION,
    images: [SOCIAL_IMAGE_PATH],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

const webApplicationJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  "@id": `${CANONICAL_URL}/#webapp`,
  name: APP_NAME,
  alternateName: "Audio to Text Converter",
  url: CANONICAL_URL,
  description: SEO_DESCRIPTION,
  applicationCategory: "MultimediaApplication",
  operatingSystem: "Web Browser",
  inLanguage: ["en", "tr"],
  keywords: SEO_KEYWORDS.join(", "),
  browserRequirements:
    "Requires a modern browser with WebAssembly support. WebGPU recommended for best performance.",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
    description: "Completely free, no hidden costs or subscriptions.",
  },
  featureList: [
    "100% Free",
    "No account required",
    "No file uploads on desktop",
    "Secure cloud fallback on mobile",
    "Private and secure",
    "Works offline after first load",
    "AI-powered by Whisper & Moonshine",
    "Mobile friendly",
    "Supports MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM, OPUS",
  ],
  potentialAction: {
    "@type": "UseAction",
    target: CANONICAL_URL,
    name: "Transcribe audio to text online",
  },
  creator: {
    "@type": "Person",
    name: "Onat Ozmen",
  },
};

const webSiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  "@id": `${CANONICAL_URL}/#website`,
  name: APP_NAME,
  url: CANONICAL_URL,
  description: SEO_DESCRIPTION,
  inLanguage: ["en", "tr"],
  publisher: {
    "@type": "Person",
    name: "Onat Ozmen",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify([webSiteJsonLd, webApplicationJsonLd]),
          }}
        />
      </head>
      <body className={`${inter.variable} bg-neutral-950 text-neutral-100 antialiased`}>
        {children}
      </body>
    </html>
  );
}
