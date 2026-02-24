import { ImageResponse } from "next/og";

export const size = {
  width: 1200,
  height: 630,
};

export const contentType = "image/png";

export const alt = "Free Audio Transcription Tool preview";

function BrandLogo() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, height: 52 }}>
      <div style={{ width: 4, height: 20, borderRadius: 99, background: "#ffffff" }} />
      <div style={{ width: 4, height: 40, borderRadius: 99, background: "#ffffff" }} />
      <div style={{ width: 4, height: 28, borderRadius: 99, background: "#ffffff" }} />
      <div style={{ width: 22, height: 4, borderRadius: 99, background: "#ffffff", marginLeft: 10 }} />
      <div style={{ width: 16, height: 4, borderRadius: 99, background: "#ffffff" }} />
    </div>
  );
}

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          padding: "56px 64px",
          color: "#f5f5f5",
          backgroundColor: "#05070a",
          backgroundImage:
            "radial-gradient(circle at 12% 20%, rgba(34,211,238,0.2), transparent 42%), radial-gradient(circle at 88% 85%, rgba(56,189,248,0.14), transparent 48%)",
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            width: "100%",
            borderRadius: 28,
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(11,15,20,0.78)",
            padding: "40px 46px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <BrandLogo />
            <div style={{ display: "flex", fontSize: 30, fontWeight: 700, color: "#f8fafc" }}>
              Audio Transcription
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 980 }}>
            <div style={{ display: "flex", fontSize: 68, lineHeight: 1.05, fontWeight: 800 }}>
              Free Audio Transcription Tool
            </div>
            <div style={{ display: "flex", fontSize: 32, color: "#cbd5e1", fontWeight: 500 }}>
              Speech to Text Online - No Sign-up
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                display: "flex",
                fontSize: 23,
                color: "#d1d5db",
                border: "1px solid rgba(255,255,255,0.16)",
                borderRadius: 999,
                padding: "8px 14px",
                background: "rgba(255,255,255,0.04)",
              }}
            >
              100% Free
            </div>
            <div
              style={{
                display: "flex",
                fontSize: 23,
                color: "#d1d5db",
                border: "1px solid rgba(255,255,255,0.16)",
                borderRadius: 999,
                padding: "8px 14px",
                background: "rgba(255,255,255,0.04)",
              }}
            >
              Supports MP3, WAV, M4A
            </div>
          </div>
        </div>
      </div>
    ),
    {
      ...size,
    },
  );
}
