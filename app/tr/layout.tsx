import type { Metadata } from "next";

const BASE_URL = "https://audio-transcription.app";
const TR_URL = `${BASE_URL}/tr`;
const SOCIAL_IMAGE_PATH = "/opengraph-image";

const TR_TITLE = "Ücretsiz Ses Transkripsiyon Aracı - Konuşmayı Metne Çevir";
const TR_DESCRIPTION =
  "Whisper ve Moonshine ile sesi ücretsiz metne çevirin. MP3, WAV, M4A ve daha fazlasını destekler. Masaüstünde yerel, mobilde güvenli bulut transkripsiyonu kullanır.";

export const metadata: Metadata = {
  title: { absolute: TR_TITLE },
  description: TR_DESCRIPTION,
  keywords: [
    "ucretsiz ses transkripsiyon",
    "sesi metne cevirme",
    "konusmayi yaziya cevirme",
    "ses kaydini metne cevir",
    "mp3 metne cevirme",
    "wav metne cevirme",
    "m4a metne cevirme",
    "ders kaydi yaziya cevirme",
    "toplanti kaydi metne cevirme",
    "whisper ai transkripsiyon",
  ],
  alternates: {
    canonical: TR_URL,
    languages: {
      en: BASE_URL,
      tr: TR_URL,
      "x-default": BASE_URL,
    },
  },
  openGraph: {
    title: TR_TITLE,
    description: TR_DESCRIPTION,
    url: TR_URL,
    locale: "tr_TR",
    alternateLocale: ["en_US"],
    type: "website",
    images: [
      {
        url: SOCIAL_IMAGE_PATH,
        width: 1200,
        height: 630,
        alt: "Ücretsiz ses transkripsiyon aracı önizlemesi",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: TR_TITLE,
    description: TR_DESCRIPTION,
    images: [SOCIAL_IMAGE_PATH],
  },
};

export default function TurkishLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return <>{children}</>;
}
