import { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://audio-transcription.app";
  const trUrl = `${baseUrl}/tr`;
  const lastModified = new Date();

  return [
    {
      url: baseUrl,
      lastModified,
      changeFrequency: "monthly",
      priority: 1,
      alternates: {
        languages: {
          en: baseUrl,
          tr: trUrl,
        },
      },
    },
    {
      url: trUrl,
      lastModified,
      changeFrequency: "monthly",
      priority: 0.95,
      alternates: {
        languages: {
          en: baseUrl,
          tr: trUrl,
        },
      },
    },
    {
      url: `${baseUrl}/privacy`,
      lastModified,
      changeFrequency: "yearly",
      priority: 0.5,
    },
  ];
}
