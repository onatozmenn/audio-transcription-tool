"use client";

import {
  AlertCircle,
  Check,
  Clock3,
  Copy,
  FileText,
  Square,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { upload } from "@vercel/blob/client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LanguageSelect } from "@/components/language-select";
import { UploadDropzone } from "@/components/upload-dropzone";
import {
  BLOB_UPLOAD_ACCESS,
  TRANSCRIPTION_SESSION_HEADER,
} from "@/lib/cloud-transcription";
import {
  encodeFloat32ToOpusOgg,
  isOpusOggEncodingSupported,
} from "@/lib/opus-encoder";
import { APP_COPY, type AppLocale } from "@/lib/app-copy";

type WhisperLanguage =
  | "english"
  | "turkish"
  | "spanish"
  | "french"
  | "german"
  | "italian"
  | "portuguese"
  | "russian"
  | "arabic"
  | "hindi"
  | "japanese"
  | "korean";

type LanguageOption = { value: WhisperLanguage; countryCode: string };
type ProgressPhase = "download" | "transcribing";
type CopyState = "idle" | "success" | "error";
type SmartExportAction = "copy_text_only" | "copy_with_timestamps" | "export_json" | "export_word" | "export_pdf";

type WorkerStatus = "loading" | "ready" | "transcribing" | "error";
type TranscriptionStatus = "idle" | "loading" | "decoding" | "transcribing" | "ready" | "error";

type WorkerRequest =
  | { type: "load"; model?: string }
  | { type: "cancel"; requestId: number }
  | {
    type: "transcribe";
    requestId: number;
    audio: Float32Array;
    language?: "auto" | WhisperLanguage;
    model?: string;
  };

type TranscriptSegment = {
  text: string;
  start: number;
  end: number;
};

type TranscriptModel = "Xenova/whisper-small" | "onnx-community/moonshine-base-ONNX" | "openai/whisper-large-v3" | null;

type TranscriptExportJson = {
  version: 1;
  createdAt: string;
  fileName: string | null;
  model: TranscriptModel;
  language: "auto" | WhisperLanguage | null;
  text: string;
  segments: TranscriptSegment[];
};

type CloudTranscribeResponse = {
  text?: string;
  segments?: Array<{ text: string; start: number; end: number }>;
};

type TranscriptionChunkGrant = {
  chunkIndex: number;
  pathname: string;
  token: string;
};

type TranscriptionSessionResponse = {
  expiresAt: string;
  grants: TranscriptionChunkGrant[];
  jobId: string;
};

type WorkerResponse =
  | { type: "status"; status: WorkerStatus; requestId?: number; detail?: string; device?: string }
  | {
    type: "progress";
    phase: ProgressPhase;
    progress: number;
    requestId?: number;
    processedChunks?: number;
    totalChunks?: number;
    currentSlice?: number;
    totalSlices?: number;
    loaded?: number;
    total?: number;
    file?: string;
  }
  | { type: "partial"; text: string; requestId: number }
  | { type: "segments"; requestId: number; text: string; segments: TranscriptSegment[] }
  | { type: "result"; text: string; requestId: number }
  | { type: "error"; error: string; requestId?: number };

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "english", countryCode: "gb" },
  { value: "turkish", countryCode: "tr" },
  { value: "spanish", countryCode: "es" },
  { value: "french", countryCode: "fr" },
  { value: "german", countryCode: "de" },
  { value: "italian", countryCode: "it" },
  { value: "portuguese", countryCode: "pt" },
  { value: "russian", countryCode: "ru" },
  { value: "arabic", countryCode: "sa" },
  { value: "hindi", countryCode: "in" },
  { value: "japanese", countryCode: "jp" },
  { value: "korean", countryCode: "kr" },
];

const TARGET_SAMPLE_RATE = 16_000;
const LOCAL_CHUNK_LENGTH_S = 30;
const LOCAL_STRIDE_LENGTH_S = 5;
const LOCAL_AUDIO_STEP_S = LOCAL_CHUNK_LENGTH_S - 2 * LOCAL_STRIDE_LENGTH_S;
// In public Blob mode, uploads go directly to Blob storage (bypassing the
// 4.5MB Vercel function body limit) and Groq fetches the URL directly.
// Larger chunks → far fewer API calls → much less chance of hitting Groq
// per-minute rate limits on long recordings.
const IS_PUBLIC_BLOB_MODE = BLOB_UPLOAD_ACCESS === "public";
const CLOUD_CHUNK_DURATION_S = IS_PUBLIC_BLOB_MODE ? 600 : 110;
// Threshold below which we upload the user's ORIGINAL (already-compressed)
// file to Blob in a single request — no decode, no WAV re-encode. This is a
// huge Blob-quota saver because typical mobile voice recordings (mp3/m4a/
// opus) are ~64-128 kbps (≈ 30-60 MB / hour), much smaller than the 16-bit
// PCM WAV we would otherwise produce (≈ 115 MB / hour).
//
// Groq whisper-large-v3 accepts up to 25 MB per request, so anything under
// MAX_AUDIO_FILE_BYTES (24 MB in public mode) can be sent as-is.
const MAX_CLOUD_DIRECT_UPLOAD_BYTES = IS_PUBLIC_BLOB_MODE
  ? 24 * 1024 * 1024
  : 3 * 1024 * 1024;
// Small compressed recordings can still represent very long audio. Once the
// duration grows beyond a normal cloud chunk, a single mobile Groq request can
// spend too long in the "uploaded, now transcribing" phase and appear stuck at
// roughly 70% progress. Force the long-file path back through chunking.
const MAX_DIRECT_CLOUD_AUDIO_SECONDS = CLOUD_CHUNK_DURATION_S;
const IOS_SAFARI_CLOUD_CHUNK_DURATION_S = 180;
const IOS_SAFARI_OPUS_FORMDATA_CHUNK_DURATION_S = 240;
const IOS_SAFARI_PCM_FORMDATA_CHUNK_DURATION_S = 90;
// Vercel serverless functions enforce a 4.5 MB request body limit.
// Keep well under that ceiling so the FormData wrapper + field names
// don't push us over.
const MAX_SAFE_FORMDATA_UPLOAD_BYTES = IS_PUBLIC_BLOB_MODE
  ? 4 * 1024 * 1024
  : 3_500_000;
const UNCOMPRESSED_CLOUD_CHUNK_DURATION_S = 180;
// 16 kHz mono PCM16 WAV bytes per second = 32 000. 10 min ≈ 19.2 MB.
// Private mode still has to proxy through the function body → keep 4 MB.
const MAX_CLOUD_CHUNK_UPLOAD_BYTES = IS_PUBLIC_BLOB_MODE
  ? 20 * 1024 * 1024
  : 4 * 1024 * 1024;
const MAX_MOBILE_DECODE_FILE_BYTES = 90 * 1024 * 1024;
const MAX_MOBILE_CLOUD_AUDIO_SECONDS = 2 * 60 * 60;
const MAX_LOCAL_DECODE_FILE_BYTES = 250 * 1024 * 1024;
const MAX_LOCAL_AUDIO_SECONDS = 2 * 60 * 60;
const MAX_CLOUD_REQUEST_RETRIES = 6;
const CLOUD_RETRY_BASE_DELAY_MS = 1_500;
const MAX_CLOUD_RETRY_DELAY_MS = 20_000;
const APP_URL = "https://audio-transcription.app";
const APP_URL_TR = `${APP_URL}/tr`;

const FAQ_JSON_LD_EN = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: [
    {
      "@type": "Question",
      name: "How can I transcribe MP3 to text for free?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Upload your MP3 file, choose language, and start transcription. The tool converts audio to text online for free without sign-up.",
      },
    },
    {
      "@type": "Question",
      name: "Do I need an account to use this speech to text tool?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "No. You can transcribe audio without creating an account.",
      },
    },
    {
      "@type": "Question",
      name: "Which audio formats are supported?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Supported formats include MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM, and OPUS.",
      },
    },
    {
      "@type": "Question",
      name: "Is my audio private during transcription?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "On desktop, transcription runs locally in your browser. On mobile, audio is uploaded temporarily for cloud transcription and deleted after processing.",
      },
    },
    {
      "@type": "Question",
      name: "Can I generate subtitle-ready transcripts?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Yes. You can export transcript text and timestamped output for subtitle and notes workflows.",
      },
    },
  ],
};

const HOW_TO_JSON_LD_EN = {
  "@context": "https://schema.org",
  "@type": "HowTo",
  name: "How to convert audio to text online",
  description:
    "Convert audio files to text in minutes using this free speech-to-text tool.",
  totalTime: "PT5M",
  step: [
    {
      "@type": "HowToStep",
      name: "Upload audio file",
      text: "Upload MP3, WAV, M4A, MP4, or another supported audio format.",
      url: `${APP_URL}/#upload-section`,
    },
    {
      "@type": "HowToStep",
      name: "Select language and start transcription",
      text: "Choose the spoken language and begin audio-to-text transcription.",
      url: `${APP_URL}/#upload-section`,
    },
    {
      "@type": "HowToStep",
      name: "Copy or export transcript",
      text: "Review transcript output, then copy plain text, timestamped text, or export JSON.",
      url: APP_URL,
    },
  ],
};

const FAQ_JSON_LD_TR = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: [
    {
      "@type": "Question",
      name: "MP3 dosyasını ücretsiz metne nasıl çevirebilirim?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "MP3 dosyanızı yükleyin, dili seçin ve transkripsiyonu başlatın. Araç sesi ücretsiz olarak metne çevirir ve üyelik gerektirmez.",
      },
    },
    {
      "@type": "Question",
      name: "Bu konuşmayı yazıya çevirme aracı için hesap gerekiyor mu?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Hayır. Hesap oluşturmadan ses dosyalarını metne çevirebilirsiniz.",
      },
    },
    {
      "@type": "Question",
      name: "Hangi ses formatları destekleniyor?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM ve OPUS formatları desteklenir.",
      },
    },
    {
      "@type": "Question",
      name: "Transkripsiyon sırasında ses verim gizli kalır mı?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Masaüstünde işlem tarayıcınızda yerel olarak yapılır. Mobilde ses güvenli bulut işleme ile metne çevrilir ve sonuç size döner.",
      },
    },
    {
      "@type": "Question",
      name: "Altyazı için kullanılabilecek çıktı alabilir miyim?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Evet. Düz metin ve zaman damgalı transcript çıktıları ile altyazı ve not alma süreçlerini destekler.",
      },
    },
  ],
};

const HOW_TO_JSON_LD_TR = {
  "@context": "https://schema.org",
  "@type": "HowTo",
  name: "Sesi çevrim içi metne çevirme adımları",
  description:
    "Bu ücretsiz konuşmayı yazıya çevirme aracıyla ses dosyalarını dakikalar içinde metne dönüştürün.",
  totalTime: "PT5M",
  step: [
    {
      "@type": "HowToStep",
      name: "Ses dosyasını yükle",
      text: "MP3, WAV, M4A, MP4 veya desteklenen başka bir ses formatını yükleyin.",
      url: `${APP_URL_TR}/#upload-section`,
    },
    {
      "@type": "HowToStep",
      name: "Dili seç ve transkripsiyonu başlat",
      text: "Konuşma dilini seçin ve sesi metne çevirme işlemini başlatın.",
      url: `${APP_URL_TR}/#upload-section`,
    },
    {
      "@type": "HowToStep",
      name: "Transcript çıktısını kopyala veya dışa aktar",
      text: "Çıktıyı kontrol edin, ardından düz metin, zaman damgalı metin veya JSON dışa aktarın.",
      url: APP_URL_TR,
    },
  ],
};

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function downmixToMono(audioBuffer: AudioBuffer): Float32Array {
  const { numberOfChannels, length } = audioBuffer;
  if (numberOfChannels === 1) {
    const mono = new Float32Array(length);
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  const mono = new Float32Array(length);
  for (let channel = 0; channel < numberOfChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      mono[i] += channelData[i];
    }
  }

  for (let i = 0; i < length; i += 1) {
    mono[i] /= numberOfChannels;
  }

  return mono;
}

function resampleMonoAudio(
  input: Float32Array,
  inputSampleRate: number,
  outputSampleRate: number,
): Float32Array {
  if (inputSampleRate === outputSampleRate) {
    return input;
  }

  const ratio = outputSampleRate / inputSampleRate;
  const outputLength = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = i / ratio;
    const lower = Math.floor(sourceIndex);
    const upper = Math.min(lower + 1, input.length - 1);
    const weight = sourceIndex - lower;
    output[i] = input[lower] * (1 - weight) + input[upper] * weight;
  }

  return output;
}

async function decodeAudioFile(file: File): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not supported in this browser.");
  }

  const audioContext = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const mono = downmixToMono(audioBuffer);
    return resampleMonoAudio(mono, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
  } finally {
    await audioContext.close();
  }
}

async function decodeAudioBufferForCloudChunking(file: File): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not supported in this browser.");
  }

  // Decode at target sample rate to reduce mobile memory pressure.
  const audioContext = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
  try {
    return await audioContext.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    await audioContext.close();
  }
}

function extractMonoChunkFromAudioBuffer(
  audioBuffer: AudioBuffer,
  startSample: number,
  endSample: number,
): Float32Array {
  const safeStart = Math.max(0, Math.min(startSample, audioBuffer.length));
  const safeEnd = Math.max(safeStart + 1, Math.min(endSample, audioBuffer.length));
  const chunkLength = safeEnd - safeStart;
  const mono = new Float32Array(chunkLength);

  if (audioBuffer.numberOfChannels === 1) {
    mono.set(audioBuffer.getChannelData(0).subarray(safeStart, safeEnd));
    return mono;
  }

  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let i = 0; i < chunkLength; i += 1) {
      mono[i] += channelData[safeStart + i];
    }
  }

  for (let i = 0; i < chunkLength; i += 1) {
    mono[i] /= audioBuffer.numberOfChannels;
  }

  return mono;
}

async function getAudioDurationSeconds(file: File): Promise<number | null> {
  const objectUrl = URL.createObjectURL(file);
  try {
    return await new Promise<number | null>((resolve) => {
      const audio = document.createElement("audio");
      audio.preload = "metadata";
      audio.src = objectUrl;

      const finish = (duration: number | null) => {
        window.clearTimeout(timeoutId);
        audio.removeEventListener("loadedmetadata", onLoaded);
        audio.removeEventListener("error", onError);
        audio.removeAttribute("src");
        resolve(duration);
      };

      const onLoaded = () => {
        const duration = audio.duration;
        if (Number.isFinite(duration) && duration > 0) {
          finish(duration);
          return;
        }
        finish(null);
      };

      const onError = () => {
        finish(null);
      };

      const timeoutId = window.setTimeout(() => finish(null), 10_000);

      audio.addEventListener("loadedmetadata", onLoaded);
      audio.addEventListener("error", onError);
      audio.load();
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // 1 channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * block align)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function timestampForFilename(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return `${year}-${month}-${day}-${hours}-${minutes}-${seconds}`;
}

function formatSegmentTimestamp(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const secs = safeSeconds % 60;

  const paddedMinutes = String(minutes).padStart(2, "0");
  const paddedSeconds = String(secs).padStart(2, "0");

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${paddedMinutes}:${paddedSeconds}`;
  }
  return `${paddedMinutes}:${paddedSeconds}`;
}

export default function Home() {
  const pathname = usePathname();
  const locale: AppLocale = pathname === "/tr" || pathname.startsWith("/tr/") ? "tr" : "en";
  const isTurkishPage = locale === "tr";
  const copy = APP_COPY[locale];
  const languageOptions = useMemo(
    () => LANGUAGE_OPTIONS.map((option) => ({
      ...option,
      label: copy.languageNames[option.value],
    })),
    [copy],
  );
  const workerRef = useRef<Worker | null>(null);
  const activeRequestIdRef = useRef(0);
  const loadedModelRef = useRef<string | null>(null);
  const copyResetTimeoutRef = useRef<number | null>(null);
  const copiedBtnTimeoutRef = useRef<number | null>(null);
  const outputTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const languageSelectRef = useRef<HTMLButtonElement | null>(null);
  const exportDialogRef = useRef<HTMLDivElement | null>(null);
  const transcribeStartedAtRef = useRef<number | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [status, setStatus] = useState<TranscriptionStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressPhase, setProgressPhase] = useState<ProgressPhase | null>(null);
  const [processedChunks, setProcessedChunks] = useState<number | null>(null);
  const [totalChunks, setTotalChunks] = useState<number | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [output, setOutput] = useState("");
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeFileName, setActiveFileName] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<"auto" | WhisperLanguage | null>(null);
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [copiedButton, setCopiedButton] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isExportTitleOpen, setIsExportTitleOpen] = useState(false);
  const [exportTitle, setExportTitle] = useState("");
  const [pendingExportAction, setPendingExportAction] = useState<"export_word" | "export_pdf" | null>(null);
  const exportTitleInputRef = useRef<HTMLInputElement | null>(null);
  const [isLangShaking, setIsLangShaking] = useState(false);
  const [justCompleted, setJustCompleted] = useState(false);
  const [downloadedBytes, setDownloadedBytes] = useState<number | null>(null);
  const [totalBytes, setTotalBytes] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"plain" | "timestamps">("plain");
  const [loadingDetail, setLoadingDetail] = useState<string | null>(null);
  const [lastTranscriptionModel, setLastTranscriptionModel] = useState<TranscriptModel>(null);
  const [currentSlice, setCurrentSlice] = useState<number | null>(null);
  const [totalSlices, setTotalSlices] = useState<number | null>(null);
  const [audioDurationSeconds, setAudioDurationSeconds] = useState<number | null>(null);
  const [warmUpElapsed, setWarmUpElapsed] = useState(0);
  const [sawActiveDownload, setSawActiveDownload] = useState(false);
  const [isMobile] = useState(
    () => typeof navigator !== "undefined" && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent),
  );
  const [isViaCloud, setIsViaCloud] = useState(false);
  const [preferFetchBlobUpload] = useState(
    () => typeof navigator !== "undefined" &&
      /iPhone|iPad|iPod/i.test(navigator.userAgent) &&
      /AppleWebKit/i.test(navigator.userAgent) &&
      !/CriOS|FxiOS|EdgiOS|OPiOS/i.test(navigator.userAgent),
  );

  // Desktop: use Moonshine (smaller/faster) for English, Whisper for other languages.
  const desiredLocalModel = useMemo(
    () => selectedLanguage === "english" ? "onnx-community/moonshine-base-ONNX" : "Xenova/whisper-small",
    [selectedLanguage],
  );
  // True if the model was successfully loaded in a previous session.
  // Stored in localStorage so page refresh doesn't re-show the loading UI.
  const [wasModelEverLoaded, setWasModelEverLoaded] = useState(
    () => typeof window !== "undefined" && localStorage.getItem("whisper_model_cached") === "1",
  );

  // Kick off model preload as soon as the user picks a language.
  // The worker's getTranscriber() caches the pipeline, so when the
  // user later drops a file the model is already ready (or still
  // downloading — in which case the transcribe request just awaits).
  // If the model is already "ready" (previously loaded), skip re-sending
  // the load message so we don't cause a brief loading flash.
  useEffect(() => {
    if (!selectedLanguage || isMobile) return;
    const worker = workerRef.current;
    if (!worker) return;
    // Send load on first language selection (idle) or when the desired model
    // changes while already ready (e.g. switching between English/other).
    if (status === "idle" || (status === "ready" && loadedModelRef.current !== desiredLocalModel)) {
      const loadRequest: WorkerRequest = { type: "load", model: desiredLocalModel };
      worker.postMessage(loadRequest);
      loadedModelRef.current = desiredLocalModel;
    }
  }, [selectedLanguage, isMobile, status, desiredLocalModel]);



  // Flash "Transcription complete" badge only when transitioning transcribing → ready
  const prevStatusRef = useRef<typeof status | null>(null);
  useEffect(() => {
    if (prevStatusRef.current === "transcribing" && status === "ready") {
      setJustCompleted(true);
      const id = window.setTimeout(() => setJustCompleted(false), 2800);
      prevStatusRef.current = status;
      return () => window.clearTimeout(id);
    }
    prevStatusRef.current = status;
  }, [status]);

  const clearProgressState = useCallback(() => {
    setProgress(0);
    setProgressPhase(null);
    setProcessedChunks(null);
    setTotalChunks(null);
    setEtaSeconds(null);
    setDownloadedBytes(null);
    setTotalBytes(null);
    setLoadingDetail(null);
    setCurrentSlice(null);
    setTotalSlices(null);
    setAudioDurationSeconds(null);
    setSawActiveDownload(false);
    setWarmUpElapsed(0);
    transcribeStartedAtRef.current = null;
  }, []);

  const handleWorkerMessage = useCallback(
    (message: WorkerResponse) => {
      if (message.type === "status") {
        if (
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }

        if (message.status === "loading") {
          setStatus("loading");
          // Only update loadingDetail for meaningful messages.
          // HuggingFace fires "initiate", "download", "progress", "done"
          // on every progress tick — those cause text jitter if shown.
          if (message.detail) {
            const noisy = ["initiate", "download", "progress", "done"];
            if (!noisy.includes(message.detail)) {
              setLoadingDetail(message.detail);
            }
          }
        } else if (message.status === "transcribing") {
          setStatus("transcribing");
          setWarmUpElapsed(0);
          setLoadingDetail(null);
          setProgress(0);
          setProgressPhase("transcribing");
        } else if (message.status === "ready") {
          setStatus("ready");
          setSawActiveDownload(false);
          setLoadingDetail(null);
          setProgressPhase(null);
          setProcessedChunks(null);
          setTotalChunks(null);
          setEtaSeconds(null);
          transcribeStartedAtRef.current = null;
          // Persist that the model has been loaded at least once so future
          // page loads can suppress the brief loading UI on reinitialisation.
          setWasModelEverLoaded(true);
          if (typeof window !== "undefined") {
            localStorage.setItem("whisper_model_cached", "1");
          }
        } else if (message.status === "error") {
          setStatus("error");
          setSawActiveDownload(false);
          setLoadingDetail(null);
          clearProgressState();
        }

        if (message.detail && message.status === "error") {
          setError(locale === "en" ? message.detail : copy.errors.workerUnexpected);
        }
        return;
      }

      if (message.type === "progress") {
        if (
          message.phase === "transcribing" &&
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }

        setProgressPhase(message.phase);

        if (message.phase === "download") {
          // Model downloads have multiple files (shards). Each file
          // independently reports 0-100% progress and its own loaded/total
          // bytes. Without guarding, the progress bar jumps backward on
          // each new file → visual jitter. Fix: never let download
          // progress or accumulated bytes decrease.
          setProgress((prev) => Math.max(prev, clampProgress(message.progress)));
          setSawActiveDownload(true);
          setStatus("loading");
          setProcessedChunks(null);
          setTotalChunks(null);
          setEtaSeconds(null);
          transcribeStartedAtRef.current = null;
          if (typeof message.loaded === "number") {
            setDownloadedBytes((prev) => Math.max(prev ?? 0, message.loaded as number));
          }
          if (typeof message.total === "number" && message.total > 0) setTotalBytes(message.total);
          return;
        }

        setProgress(clampProgress(message.progress));

        setStatus("transcribing");
        setProcessedChunks(message.processedChunks ?? null);
        setTotalChunks(message.totalChunks ?? null);
        if (typeof message.currentSlice === "number") setCurrentSlice(message.currentSlice);
        if (typeof message.totalSlices === "number") setTotalSlices(message.totalSlices);

        if (transcribeStartedAtRef.current === null) {
          transcribeStartedAtRef.current = Date.now();
        }

        const processed = message.processedChunks;
        const total = message.totalChunks;
        if (
          typeof processed === "number" &&
          typeof total === "number" &&
          processed > 0 &&
          total >= processed
        ) {
          const elapsedSeconds = (Date.now() - transcribeStartedAtRef.current) / 1000;
          const averageChunkSeconds = elapsedSeconds / processed;
          const remainingChunks = Math.max(total - processed, 0);
          const estimatedRemaining = Math.ceil(averageChunkSeconds * remainingChunks);
          setEtaSeconds(estimatedRemaining);
        } else {
          setEtaSeconds(null);
        }
        return;
      }

      if (message.type === "partial") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setStatus("transcribing");
        setOutput(message.text);
        return;
      }

      if (message.type === "segments") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setSegments(message.segments);
        setOutput(message.text);
        setStatus("ready");
        setProgressPhase(null);
        setProcessedChunks(null);
        setTotalChunks(null);
        setEtaSeconds(null);
        transcribeStartedAtRef.current = null;
        return;
      }

      if (message.type === "result") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setOutput(message.text);
        setStatus("ready");
        setProgressPhase(null);
        setProcessedChunks(null);
        setTotalChunks(null);
        setEtaSeconds(null);
        transcribeStartedAtRef.current = null;
        return;
      }

      if (message.type === "error") {
        if (
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }
        setStatus("error");
        setError(locale === "en" ? message.error : copy.errors.workerUnexpected);
        clearProgressState();
      }
    },
    [clearProgressState, copy, locale],
  );

  const initializeWorker = useCallback(() => {
    // Mobile uses cloud transcription — skip the heavy worker + model download.
    if (typeof navigator !== "undefined" && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
      return null;
    }

    if (typeof Worker === "undefined") {
      queueMicrotask(() => {
        setStatus("error");
        setError(copy.errors.workersUnsupported);
      });
      return null;
    }

    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      handleWorkerMessage(event.data);
    };

    worker.onerror = (event) => {
      setStatus("error");
      setError(locale === "en" && event.message ? event.message : copy.errors.workerUnexpected);
      clearProgressState();
    };

    return worker;
  }, [clearProgressState, copy, handleWorkerMessage, locale]);

  const cancelTranscription = useCallback(() => {
    const currentRequestId = activeRequestIdRef.current;
    activeRequestIdRef.current += 1;
    setIsCancelling(true);
    setError(null);
    clearProgressState();

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsViaCloud(false);
      setStatus("ready");
      setActiveFileName(null);
      // Cloud operation cancelled — no worker action needed.
      window.setTimeout(() => setIsCancelling(false), 300);
      return;
    }

    const isModelLoaded =
      status === "ready" || status === "transcribing" || status === "decoding";

    if (isModelLoaded && !isViaCloud) {
      // Model is already in memory — just abort the running transcription.
      // No need to kill the worker; the loaded pipeline stays intact.
      if (workerRef.current) {
        const cancelRequest: WorkerRequest = { type: "cancel", requestId: currentRequestId };
        (workerRef.current as Worker).postMessage(cancelRequest);
      }
      setStatus("ready");
      setActiveFileName(null);
    } else {
      // Model wasn't loaded yet — terminate and let it restart cleanly.
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
      const newWorker = initializeWorker();
      if (selectedLanguage && newWorker) {
        const loadRequest: WorkerRequest = { type: "load", model: desiredLocalModel };
        newWorker.postMessage(loadRequest);
        loadedModelRef.current = desiredLocalModel;
      }
      setStatus("idle");
    }

    window.setTimeout(() => {
      setIsCancelling(false);
    }, 300);
  }, [clearProgressState, initializeWorker, isViaCloud, selectedLanguage, status, desiredLocalModel]);

  useEffect(() => {
    initializeWorker();

    return () => {
      if (copyResetTimeoutRef.current !== null) {
        window.clearTimeout(copyResetTimeoutRef.current);
      }
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [initializeWorker]);

  useEffect(() => {
    const textarea = outputTextareaRef.current;
    if (!textarea) return;

    textarea.style.height = "320px";
    textarea.style.overflowY = "auto";

    // Auto-scroll to the bottom while transcription is streaming so the user
    // can watch new chunks appear live without manually scrolling.
    if (status === "transcribing") {
      textarea.scrollTop = textarea.scrollHeight;
    }
  }, [output, status]);

  useEffect(() => {
    if (progressPhase !== "transcribing" || etaSeconds === null || etaSeconds <= 0) {
      return;
    }

    const countdown = window.setInterval(() => {
      setEtaSeconds((current) => {
        if (current === null || current <= 0) return 0;
        return current - 1;
      });
    }, 1000);

    return () => {
      window.clearInterval(countdown);
    };
  }, [etaSeconds, progressPhase]);

  const startTranscription = useCallback(
    async (file: File) => {
      // Don't cancel/restart the worker if we're only preloading the model
      // (status === "loading" with no active file). Only cancel if a previous
      // transcription is already in flight.
      if (status === "transcribing" || status === "decoding") {
        cancelTranscription();
      }

      if (!isMobile) {
        const worker = workerRef.current;
        if (!worker) {
          setStatus("error");
          setError(copy.errors.workerUnavailable);
          return;
        }
      }

      const requestId = activeRequestIdRef.current + 1;
      activeRequestIdRef.current = requestId;

      setActiveFileName(file.name);
      setOutput("");
      setSegments([]);
      setViewMode("plain");
      setCopyState("idle");
      setCopyFeedback(null);
      setError(null);
      setLastTranscriptionModel(null);
      clearProgressState();

      if (isMobile) {
        setIsViaCloud(true);
        setLastTranscriptionModel("openai/whisper-large-v3");
        setStatus("transcribing");
        setProgressPhase("transcribing");
        setLoadingDetail(copy.uploadToCloud);
        setProgress(0);

        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
          const parseRetryAfterSeconds = (retryAfterHeader: string | null): number | null => {
            if (!retryAfterHeader) return null;

            const seconds = Number.parseInt(retryAfterHeader, 10);
            if (Number.isFinite(seconds) && seconds > 0) {
              return seconds;
            }

            const parsedDate = Date.parse(retryAfterHeader);
            if (Number.isNaN(parsedDate)) return null;

            const deltaSeconds = Math.ceil((parsedDate - Date.now()) / 1_000);
            return deltaSeconds > 0 ? deltaSeconds : null;
          };

          const retryDelayMs = (attempt: number, retryAfterHeader: string | null): number => {
            const backoffMs = Math.min(
              MAX_CLOUD_RETRY_DELAY_MS,
              CLOUD_RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempt - 1),
            );
            const retryAfterSeconds = parseRetryAfterSeconds(retryAfterHeader);
            if (retryAfterSeconds === null) return backoffMs;
            return Math.max(backoffMs, retryAfterSeconds * 1_000);
          };

          const waitWithAbort = (ms: number): Promise<void> =>
            new Promise((resolve, reject) => {
              if (controller.signal.aborted) {
                reject(new DOMException("Aborted", "AbortError"));
                return;
              }

              const timeoutId = window.setTimeout(() => {
                controller.signal.removeEventListener("abort", onAbort);
                resolve();
              }, ms);

              const onAbort = () => {
                window.clearTimeout(timeoutId);
                controller.signal.removeEventListener("abort", onAbort);
                reject(new DOMException("Aborted", "AbortError"));
              };

              controller.signal.addEventListener("abort", onAbort, { once: true });
            });

          const createCloudSession = async (
            totalChunks: number,
            sessionFileName: string,
          ): Promise<TranscriptionSessionResponse> => {
            const maxAttempts = 3;

            for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
              let response: Response;
              try {
                response = await fetch("/api/transcribe/session", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    fileName: sessionFileName,
                    totalChunks,
                  }),
                  signal: controller.signal,
                });
              } catch (requestError) {
                if (requestError instanceof DOMException && requestError.name === "AbortError") {
                  throw requestError;
                }

                if (attempt >= maxAttempts) throw requestError;

                const delayMs = retryDelayMs(attempt, null);
                setLoadingDetail(copy.secureSessionRetry(Math.ceil(delayMs / 1_000)));
                await waitWithAbort(delayMs);
                continue;
              }

              if (response.ok) {
                return (await response.json()) as TranscriptionSessionResponse;
              }

              const hasAttemptsLeft = attempt < maxAttempts;
              const shouldRetry = response.status === 429 || response.status >= 500;

              if (shouldRetry && hasAttemptsLeft) {
                const delayMs = retryDelayMs(attempt, response.headers.get("retry-after"));
                setLoadingDetail(copy.secureSessionBusy(Math.ceil(delayMs / 1_000)));
                await waitWithAbort(delayMs);
                continue;
              }

              throw new Error(copy.errors.cloudRequestFailed(response.status));
            }

            throw new Error(copy.errors.sessionStartFailed);
          };

          const cleanupUploadedCloudBlob = async (
            blobUrl: string,
            chunkGrant: TranscriptionChunkGrant,
          ): Promise<void> => {
            for (let attempt = 1; attempt <= 3; attempt += 1) {
              try {
                const response = await fetch("/api/blob/delete", {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    [TRANSCRIPTION_SESSION_HEADER]: chunkGrant.token,
                  },
                  body: JSON.stringify({ blobUrl }),
                });
                if (response.ok) return;
              } catch {
                // Retry below; the scheduled cleanup route is the final fallback.
              }

              if (attempt < 3) {
                await new Promise((resolve) => {
                  window.setTimeout(resolve, 500 * 2 ** (attempt - 1));
                });
              }
            }
          };

          const uploadCloudBlob = async (
            blob: Blob,
            chunkGrant: TranscriptionChunkGrant,
            chunkIndex: number,
            totalChunks: number,
          ) => {
            // Baseline progress at the start of this chunk's upload phase.
            const uploadFloor = (chunkIndex / totalChunks) * 100;
            // Upload is considered ~70% of a chunk's work; transcription
            // (Groq fetch) is the remaining ~30%.
            const uploadCeiling = ((chunkIndex + 0.7) / totalChunks) * 100;
            // @vercel/blob multipart uploads emit per-part progress that can
            // appear to reset between parts. Guard against visual oscillation
            // by only ever increasing the progress value within this chunk.
            let highestSeenPercentage = 0;

            // Some mobile Safari builds never emit onUploadProgress events
            // for the @vercel/blob client. Creep the bar linearly based on
            // elapsed time so the user sees motion even when the browser
            // stays silent. Caps at the upload ceiling so a real progress
            // event can always overtake it.
            const creepStartMs = Date.now();
            // Assume a conservative 2 Mbps effective upload speed so a
            // 20 MB chunk creeps to ~90% of the upload ceiling over ~80s.
            const creepDurationMs = Math.max(20_000, (blob.size / (2 * 125_000)) * 1_000);
            const creepHandle = window.setInterval(() => {
              if (requestId !== activeRequestIdRef.current) return;
              const elapsed = Date.now() - creepStartMs;
              const fraction = Math.min(0.9, elapsed / creepDurationMs);
              const creepTarget = uploadFloor + (uploadCeiling - uploadFloor) * fraction;
              setProgress((prev) => (creepTarget > prev ? creepTarget : prev));
            }, 400);

            // Timeout guard: if the upload promise never resolves (mobile
            // Safari + @vercel/blob multipart edge case), force-abort after
            // a generous timeout so we fall into the retry loop instead of
            // hanging forever.
            const uploadTimeoutController = new AbortController();
            const uploadTimeoutMs = Math.min(
              10 * 60_000,
              Math.max(
                180_000,
                Math.ceil((blob.size / 100_000) * 1_000) + 30_000,
              ),
            );
            const uploadTimeoutId = window.setTimeout(() => {
              uploadTimeoutController.abort();
            }, uploadTimeoutMs);
            const onOuterAbort = () => uploadTimeoutController.abort();
            controller.signal.addEventListener("abort", onOuterAbort, { once: true });

            try {
              const uploadOptions: Parameters<typeof upload>[2] = {
                access: BLOB_UPLOAD_ACCESS,
                handleUploadUrl: "/api/blob/upload",
                contentType: blob.type || undefined,
                // Force single-part PUT. @vercel/blob's automatic multipart
                // path has been observed to hang on mobile Safari (the
                // "complete multipart" promise occasionally never resolves
                // client-side even though the server-side upload succeeded).
                multipart: false,
                headers: {
                  [TRANSCRIPTION_SESSION_HEADER]: chunkGrant.token,
                },
                abortSignal: uploadTimeoutController.signal,
                ...(preferFetchBlobUpload
                  ? {}
                  : {
                    onUploadProgress: ({ percentage }: { percentage: number }) => {
                      if (requestId !== activeRequestIdRef.current) return;
                      if (percentage > highestSeenPercentage) {
                        highestSeenPercentage = percentage;
                      }
                      const chunkProgress =
                        uploadFloor + (uploadCeiling - uploadFloor) * (highestSeenPercentage / 100);
                      setProgress((prev) => (chunkProgress > prev ? chunkProgress : prev));
                    },
                  }),
              };

              return await upload(chunkGrant.pathname, blob, uploadOptions);
            } finally {
              window.clearInterval(creepHandle);
              window.clearTimeout(uploadTimeoutId);
              controller.signal.removeEventListener("abort", onOuterAbort);
            }
          };

          const transcribeCloudBlob = async (
            blob: Blob,
            chunkGrant: TranscriptionChunkGrant,
            chunkIndex: number,
            totalChunks: number,
          ): Promise<CloudTranscribeResponse> => {
            const maxAttempts = MAX_CLOUD_REQUEST_RETRIES + 1;

            // Retry the upload itself (not just the transcribe call).
            // Mobile Safari + @vercel/blob occasionally hang mid-upload;
            // the inner uploadCloudBlob has its own 120s abort guard, so
            // here we just retry with backoff when that guard fires.
            let uploadedBlob: { url: string } | null = null;
            for (let uploadAttempt = 1; uploadAttempt <= maxAttempts; uploadAttempt += 1) {
              try {
                uploadedBlob = await uploadCloudBlob(
                  blob,
                  chunkGrant,
                  chunkIndex,
                  totalChunks,
                );
                break;
              } catch (uploadError) {
                if (controller.signal.aborted) {
                  throw new DOMException("Aborted", "AbortError");
                }
                const hasAttemptsLeft = uploadAttempt < maxAttempts;
                if (!hasAttemptsLeft) throw uploadError;
                const delayMs = retryDelayMs(uploadAttempt, null);
                setLoadingDetail(copy.uploadRetry(Math.ceil(delayMs / 1_000)));
                await waitWithAbort(delayMs);
              }
            }
            if (!uploadedBlob) {
              throw new Error(copy.errors.cloudUploadFailed);
            }

            try {
              for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
                let response: Response;
                // Visually creep the progress bar from the upload-finished
                // floor up toward (but not reaching) the chunk-complete
                // ceiling while waiting for Groq. Prevents the bar from
                // looking frozen during the 10-40s transcription call.
                const transcribeFloor = ((chunkIndex + 0.7) / totalChunks) * 100;
                const transcribeCeiling = ((chunkIndex + 0.97) / totalChunks) * 100;
                const creepStartMs = Date.now();
                const creepHandle = window.setInterval(() => {
                  if (requestId !== activeRequestIdRef.current) return;
                  // Asymptote at transcribeCeiling over ~45s.
                  const elapsed = (Date.now() - creepStartMs) / 1_000;
                  const t = 1 - Math.exp(-elapsed / 15);
                  const next = transcribeFloor + (transcribeCeiling - transcribeFloor) * t;
                  setProgress((prev) => (next > prev ? next : prev));
                }, 400);
                // Client-side timeout guard. /api/transcribe has a 60s
                // serverless maxDuration; give the fetch an extra ~30s
                // margin before we assume the request is stuck and retry.
                const fetchTimeoutController = new AbortController();
                const onOuterAbort = () => fetchTimeoutController.abort();
                controller.signal.addEventListener("abort", onOuterAbort, { once: true });
                const fetchTimeoutId = window.setTimeout(() => {
                  fetchTimeoutController.abort();
                }, 90_000);

                try {
                  setLoadingDetail(
                    totalChunks > 1
                      ? copy.transcribingChunk(chunkIndex + 1, totalChunks)
                      : copy.transcribingAudio,
                  );

                  response = await fetch("/api/transcribe", {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                      [TRANSCRIPTION_SESSION_HEADER]: chunkGrant.token,
                    },
                    body: JSON.stringify({
                      blobUrl: uploadedBlob.url,
                      language: selectedLanguage ?? "auto",
                    }),
                    signal: fetchTimeoutController.signal,
                  });
                } catch (requestError) {
                  // If the outer user cancel fired, propagate as abort.
                  if (controller.signal.aborted) {
                    throw new DOMException("Aborted", "AbortError");
                  }

                  const hasAttemptsLeft = attempt < maxAttempts;
                  if (!hasAttemptsLeft) throw requestError;

                  const delayMs = retryDelayMs(attempt, null);
                  setLoadingDetail(copy.cloudConnectionRetry(Math.ceil(delayMs / 1_000)));
                  await waitWithAbort(delayMs);
                  continue;
                } finally {
                  window.clearInterval(creepHandle);
                  window.clearTimeout(fetchTimeoutId);
                  controller.signal.removeEventListener("abort", onOuterAbort);
                }

                if (response.ok) {
                  return (await response.json()) as CloudTranscribeResponse;
                }

                const hasAttemptsLeft = attempt < maxAttempts;
                const shouldRetry = response.status === 429 || response.status >= 500;

                if (shouldRetry && hasAttemptsLeft) {
                  const retryAfterHeader = response.headers.get("retry-after");
                  const delayMs = retryDelayMs(
                    attempt,
                    retryAfterHeader,
                  );
                  const effectiveDelayMs =
                    response.status === 429
                      ? Math.max(
                        delayMs,
                        retryAfterHeader ? 0 : Math.min(60_000, 20_000 * attempt),
                      )
                      : delayMs;
                  setLoadingDetail(
                    response.status === 429
                      ? copy.cloudRateLimitRetry(Math.ceil(effectiveDelayMs / 1_000))
                      : copy.cloudBusyRetry(Math.ceil(effectiveDelayMs / 1_000)),
                  );
                  await waitWithAbort(effectiveDelayMs);
                  continue;
                }

                throw new Error(copy.errors.cloudRequestFailed(response.status));
              }

              throw new Error(copy.errors.cloudTranscriptionFailed);
            } finally {
              await cleanupUploadedCloudBlob(uploadedBlob.url, chunkGrant);
            }
          };

          const transcribeDirectFormDataBlob = async (
            blob: Blob,
            chunkIndex: number,
            totalChunks: number,
            fileName: string,
            sessionToken: string,
          ): Promise<CloudTranscribeResponse> => {
            const maxAttempts = MAX_CLOUD_REQUEST_RETRIES + 1;

            for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
              let response: Response;
              const chunkFloor = (chunkIndex / totalChunks) * 100;
              const chunkCeiling = ((chunkIndex + 0.97) / totalChunks) * 100;
              const creepStartMs = Date.now();
              const creepHandle = window.setInterval(() => {
                if (requestId !== activeRequestIdRef.current) return;
                const elapsed = (Date.now() - creepStartMs) / 1_000;
                const t = 1 - Math.exp(-elapsed / 15);
                const next = chunkFloor + (chunkCeiling - chunkFloor) * t;
                setProgress((prev) => (next > prev ? next : prev));
              }, 400);

              const fetchTimeoutController = new AbortController();
              const onOuterAbort = () => fetchTimeoutController.abort();
              controller.signal.addEventListener("abort", onOuterAbort, { once: true });
              const fetchTimeoutId = window.setTimeout(() => {
                fetchTimeoutController.abort();
              }, 90_000);

              try {
                const formData = new FormData();
                formData.append("file", blob, fileName);
                formData.append("language", selectedLanguage ?? "auto");

                setLoadingDetail(
                  totalChunks > 1
                    ? copy.transcribingChunk(chunkIndex + 1, totalChunks)
                    : copy.transcribingAudio,
                );

                response = await fetch("/api/transcribe", {
                  method: "POST",
                  headers: {
                    [TRANSCRIPTION_SESSION_HEADER]: sessionToken,
                  },
                  body: formData,
                  signal: fetchTimeoutController.signal,
                });
              } catch (requestError) {
                if (controller.signal.aborted) {
                  throw new DOMException("Aborted", "AbortError");
                }

                const hasAttemptsLeft = attempt < maxAttempts;
                if (!hasAttemptsLeft) throw requestError;

                const delayMs = retryDelayMs(attempt, null);
                setLoadingDetail(copy.cloudConnectionRetry(Math.ceil(delayMs / 1_000)));
                await waitWithAbort(delayMs);
                continue;
              } finally {
                window.clearInterval(creepHandle);
                window.clearTimeout(fetchTimeoutId);
                controller.signal.removeEventListener("abort", onOuterAbort);
              }

              if (response.ok) {
                return (await response.json()) as CloudTranscribeResponse;
              }

              const hasAttemptsLeft = attempt < maxAttempts;
              const shouldRetry = response.status === 429 || response.status >= 500;

              if (shouldRetry && hasAttemptsLeft) {
                const retryAfterHeader = response.headers.get("retry-after");
                const delayMs = retryDelayMs(
                  attempt,
                  retryAfterHeader,
                );
                const effectiveDelayMs =
                  response.status === 429
                    ? Math.max(
                      delayMs,
                      retryAfterHeader ? 0 : Math.min(60_000, 20_000 * attempt),
                    )
                    : delayMs;
                setLoadingDetail(
                  response.status === 429
                    ? copy.cloudRateLimitRetry(Math.ceil(effectiveDelayMs / 1_000))
                    : copy.cloudBusyRetry(Math.ceil(effectiveDelayMs / 1_000)),
                );
                await waitWithAbort(effectiveDelayMs);
                continue;
              }

              throw new Error(copy.errors.cloudRequestFailed(response.status));
            }

            throw new Error(copy.errors.cloudTranscriptionFailed);
          };

          // Small mobile files can upload directly. Larger ones are decoded once,
          // then transcoded and sent to Blob storage chunk-by-chunk.
          let totalCloudChunks = 1;
          let decodedAudioBuffer: AudioBuffer | null = null;
          let sourceSampleRate = TARGET_SAMPLE_RATE;
          let sourceSamplesPerChunk = 0;
          let targetSamplesPerChunk = 0;
          const useSafariDirectTranscribe = preferFetchBlobUpload;
          const useOpusEncoding =
            isOpusOggEncodingSupported() && !useSafariDirectTranscribe;
          const canAttemptSingleBlobUpload = useSafariDirectTranscribe
            ? file.size <= MAX_SAFE_FORMDATA_UPLOAD_BYTES
            : file.size <= MAX_CLOUD_DIRECT_UPLOAD_BYTES;
          let detectedDurationSeconds: number | null = null;

          if (useSafariDirectTranscribe) {
            void getAudioDurationSeconds(file).then((duration) => {
              if (requestId !== activeRequestIdRef.current) return;
              if (duration === null) return;
              setAudioDurationSeconds(Math.max(1, Math.round(duration)));
            });
          } else if (canAttemptSingleBlobUpload) {
            detectedDurationSeconds = await getAudioDurationSeconds(file);
            if (requestId !== activeRequestIdRef.current) return;

            if (detectedDurationSeconds !== null) {
              setAudioDurationSeconds(Math.max(1, Math.round(detectedDurationSeconds)));

              if (detectedDurationSeconds > MAX_MOBILE_CLOUD_AUDIO_SECONDS) {
                throw new Error(copy.errors.mobileTooLong);
              }
            }
          }

          const shouldUseSingleBlobUpload =
            useSafariDirectTranscribe
              ? canAttemptSingleBlobUpload
              : canAttemptSingleBlobUpload &&
                (detectedDurationSeconds === null ||
                  detectedDurationSeconds <= MAX_DIRECT_CLOUD_AUDIO_SECONDS);

          if (!shouldUseSingleBlobUpload) {
            if (file.size > MAX_MOBILE_DECODE_FILE_BYTES) {
              throw new Error(copy.errors.mobileTooLarge);
            }

            setLoadingDetail(
              canAttemptSingleBlobUpload
                ? copy.longRecording
                : copy.preparingUpload,
            );
            setStatus("decoding"); // visually update
            const audioBuffer = await decodeAudioBufferForCloudChunking(file);
            if (requestId !== activeRequestIdRef.current) return;

            setStatus("transcribing"); // back to transcribing
            const audioDuration = audioBuffer.length / audioBuffer.sampleRate;
            if (audioDuration > MAX_MOBILE_CLOUD_AUDIO_SECONDS) {
              throw new Error(copy.errors.mobileTooLong);
            }
            setAudioDurationSeconds(Math.max(1, Math.round(audioDuration)));

            // Keep each WAV chunk safely below the server body limit.
            // When WebCodecs Opus is available we compress ~10× on the fly,
            // so the per-chunk size budget is dominated by audio duration
            // rather than raw PCM bytes.
            const preferredChunkDurationSeconds = Math.min(
              useSafariDirectTranscribe
                ? (useOpusEncoding
                  ? IOS_SAFARI_OPUS_FORMDATA_CHUNK_DURATION_S
                  : IOS_SAFARI_PCM_FORMDATA_CHUNK_DURATION_S)
                : CLOUD_CHUNK_DURATION_S,
              preferFetchBlobUpload && !useSafariDirectTranscribe
                ? IOS_SAFARI_CLOUD_CHUNK_DURATION_S
                : useOpusEncoding
                  ? CLOUD_CHUNK_DURATION_S
                  : UNCOMPRESSED_CLOUD_CHUNK_DURATION_S,
            );
            const desiredTargetSamplesPerChunk =
              preferredChunkDurationSeconds * TARGET_SAMPLE_RATE;
            const maxTargetSamplesPerChunk = useOpusEncoding
              ? desiredTargetSamplesPerChunk
              : Math.max(1, Math.floor((MAX_CLOUD_CHUNK_UPLOAD_BYTES - 44) / 2));
            targetSamplesPerChunk = Math.max(
              1,
              Math.min(desiredTargetSamplesPerChunk, maxTargetSamplesPerChunk),
            );
            decodedAudioBuffer = audioBuffer;
            sourceSampleRate = audioBuffer.sampleRate;
            sourceSamplesPerChunk = Math.max(
              1,
              Math.round((targetSamplesPerChunk * sourceSampleRate) / TARGET_SAMPLE_RATE),
            );
            totalCloudChunks = Math.max(
              1,
              Math.ceil(audioBuffer.length / sourceSamplesPerChunk),
            );
          }
          setTotalChunks(totalCloudChunks);

          if (useSafariDirectTranscribe) {
            // Create a session so the x-transcription-session header is
            // present — Vercel edge bot protection requires it.
            setLoadingDetail(copy.startingSecureSession);
            const directSessionFileName =
              decodedAudioBuffer !== null && useOpusEncoding ? "chunk.ogg" : file.name;
            const directSession = await createCloudSession(
              totalCloudChunks,
              directSessionFileName,
            );
            if (requestId !== activeRequestIdRef.current) return;
            if (directSession.grants.length !== totalCloudChunks) {
              throw new Error(copy.errors.sessionGrantMissing);
            }

            let combinedText = "";
            const combinedSegments: TranscriptSegment[] = [];

            for (let i = 0; i < totalCloudChunks; i++) {
              if (abortControllerRef.current?.signal.aborted) break;

              const isChunkedUpload = decodedAudioBuffer !== null;
              let uploadBlob = file as Blob;
              let offsetS = 0;
              let uploadFileName = file.name;

              if (isChunkedUpload && decodedAudioBuffer) {
                const chunkStartSample = i * sourceSamplesPerChunk;
                const chunkEndSample = Math.min(
                  chunkStartSample + sourceSamplesPerChunk,
                  decodedAudioBuffer.length,
                );
                let chunkData = extractMonoChunkFromAudioBuffer(
                  decodedAudioBuffer,
                  chunkStartSample,
                  chunkEndSample,
                );
                if (sourceSampleRate !== TARGET_SAMPLE_RATE) {
                  chunkData = resampleMonoAudio(
                    chunkData,
                    sourceSampleRate,
                    TARGET_SAMPLE_RATE,
                  );
                }
                if (chunkData.length > targetSamplesPerChunk) {
                  chunkData = chunkData.slice(0, targetSamplesPerChunk);
                }

                if (useOpusEncoding) {
                  try {
                    uploadBlob = await encodeFloat32ToOpusOgg(
                      chunkData,
                      TARGET_SAMPLE_RATE,
                    );
                    uploadFileName = "chunk.ogg";
                  } catch (opusError) {
                    console.warn(
                      "Opus encoding failed, falling back to WAV:",
                      opusError,
                    );
                    uploadBlob = encodeWAV(chunkData, TARGET_SAMPLE_RATE);
                    uploadFileName = "chunk.wav";
                  }
                } else {
                  uploadBlob = encodeWAV(chunkData, TARGET_SAMPLE_RATE);
                  uploadFileName = "chunk.wav";
                }

                offsetS = chunkStartSample / sourceSampleRate;
              }

              setLoadingDetail(
                totalCloudChunks > 1
                  ? copy.preparingChunk(i + 1, totalCloudChunks)
                  : copy.preparingAudio,
              );
              setProcessedChunks(i);
              setProgress((prev) => {
                const next = (i / totalCloudChunks) * 100;
                return next > prev ? next : prev;
              });

              if (requestId !== activeRequestIdRef.current) return;
              const result = await transcribeDirectFormDataBlob(
                uploadBlob,
                i,
                totalCloudChunks,
                uploadFileName,
                directSession.grants[i].token,
              );

              setProcessedChunks(i + 1);
              setProgress((prev) => {
                const next = ((i + 1) / totalCloudChunks) * 100;
                return next > prev ? next : prev;
              });

              combinedText += (combinedText ? " " : "") + (result.text || "").trim();

              if (Array.isArray(result.segments)) {
                for (const seg of result.segments) {
                  combinedSegments.push({
                    text: seg.text,
                    start: seg.start + offsetS,
                    end: seg.end + offsetS,
                  });
                }
              }
            }

            if (abortControllerRef.current?.signal.aborted) return;
            if (requestId !== activeRequestIdRef.current) return;

            setProgress(100);
            setProcessedChunks(totalCloudChunks);
            setOutput(combinedText);
            setSegments(combinedSegments);
            setStatus("ready");
            setProgressPhase(null);
            setIsViaCloud(false);
            abortControllerRef.current = null;
            return;
          }

          setLoadingDetail(copy.startingSecureSession);
          // When we transcode chunks to Opus, the session-reserved pathname
          // MUST end in .ogg so Groq's URL-based upload treats the payload
          // as Ogg/Opus rather than guessing from the original extension.
          const isChunkedCloudUpload = decodedAudioBuffer !== null;
          const sessionFileName =
            isChunkedCloudUpload && useOpusEncoding ? "chunk.ogg" : file.name;
          const cloudSession = await createCloudSession(
            totalCloudChunks,
            sessionFileName,
          );
          if (requestId !== activeRequestIdRef.current) return;
          if (cloudSession.grants.length !== totalCloudChunks) {
            throw new Error(copy.errors.sessionGrantMissing);
          }

          let combinedText = "";
          const combinedSegments: TranscriptSegment[] = [];

          for (let i = 0; i < totalCloudChunks; i++) {
            if (abortControllerRef.current?.signal.aborted) break;

            const isChunkedUpload = decodedAudioBuffer !== null;
            let uploadBlob = file as Blob;
            let offsetS = 0;

            if (isChunkedUpload && decodedAudioBuffer) {
              const chunkStartSample = i * sourceSamplesPerChunk;
              const chunkEndSample = Math.min(
                chunkStartSample + sourceSamplesPerChunk,
                decodedAudioBuffer.length,
              );
              let chunkData = extractMonoChunkFromAudioBuffer(
                decodedAudioBuffer,
                chunkStartSample,
                chunkEndSample,
              );
              if (sourceSampleRate !== TARGET_SAMPLE_RATE) {
                chunkData = resampleMonoAudio(
                  chunkData,
                  sourceSampleRate,
                  TARGET_SAMPLE_RATE,
                );
              }
              if (chunkData.length > targetSamplesPerChunk) {
                chunkData = chunkData.slice(0, targetSamplesPerChunk);
              }
              // Prefer Opus (≈10× smaller than PCM WAV). Fall back to WAV
              // if the browser lacks WebCodecs or the encoder throws at
              // runtime — either way the resulting Blob is accepted by Groq.
              if (useOpusEncoding) {
                try {
                  uploadBlob = await encodeFloat32ToOpusOgg(
                    chunkData,
                    TARGET_SAMPLE_RATE,
                  );
                } catch (opusError) {
                  console.warn(
                    "Opus encoding failed, falling back to WAV:",
                    opusError,
                  );
                  uploadBlob = encodeWAV(chunkData, TARGET_SAMPLE_RATE);
                }
              } else {
                uploadBlob = encodeWAV(chunkData, TARGET_SAMPLE_RATE);
              }
              offsetS = chunkStartSample / sourceSampleRate;
            }

            setLoadingDetail(
              totalCloudChunks > 1
                ? copy.uploadingChunk(i + 1, totalCloudChunks)
                : copy.uploadingSecurely,
            );
            setProcessedChunks(i);
            setProgress((prev) => {
              const next = (i / totalCloudChunks) * 100;
              return next > prev ? next : prev;
            });

            if (requestId !== activeRequestIdRef.current) return;
            const chunkGrant = cloudSession.grants[i];
            if (!chunkGrant || chunkGrant.chunkIndex !== i) {
              throw new Error(copy.errors.sessionGrantMissing);
            }

            const result = await transcribeCloudBlob(
              uploadBlob,
              chunkGrant,
              i,
              totalCloudChunks,
            );
            setProcessedChunks(i + 1);
            setProgress((prev) => {
              const next = ((i + 1) / totalCloudChunks) * 100;
              return next > prev ? next : prev;
            });

            combinedText += (combinedText ? " " : "") + (result.text || "").trim();

            if (Array.isArray(result.segments)) {
              for (const seg of result.segments) {
                combinedSegments.push({
                  text: seg.text,
                  start: seg.start + offsetS,
                  end: seg.end + offsetS,
                });
              }
            }
          }

          if (abortControllerRef.current?.signal.aborted) return;
          if (requestId !== activeRequestIdRef.current) return;

          setProgress(100);
          setProcessedChunks(totalCloudChunks);
          setOutput(combinedText);
          setSegments(combinedSegments);
          setStatus("ready");
          setProgressPhase(null);
          setIsViaCloud(false);
          abortControllerRef.current = null;

        } catch (cloudError: unknown) {
          if (controller.signal.aborted) return;
          if (requestId !== activeRequestIdRef.current) return;
          setStatus("error");
          const cloudMessage = cloudError instanceof Error ? cloudError.message : "";
          const localizedCloudErrors = new Set([
            copy.errors.cloudRequestFailed(400),
            copy.errors.cloudTranscriptionFailed,
            copy.errors.cloudUploadFailed,
            copy.errors.mobileTooLarge,
            copy.errors.mobileTooLong,
            copy.errors.sessionGrantMissing,
            copy.errors.sessionStartFailed,
          ]);
          const localizedHttpPrefix = copy.errors.cloudRequestFailed(0).split("0")[0];
          const isLocalizedHttpError = cloudMessage.startsWith(localizedHttpPrefix);
          setError(
            locale === "en" || localizedCloudErrors.has(cloudMessage) || isLocalizedHttpError
              ? cloudMessage || copy.errors.cloudTranscriptionFailed
              : copy.errors.cloudTranscriptionFailed,
          );
          setIsViaCloud(false);
          abortControllerRef.current = null;
        }
        return;
      }

      setStatus("decoding");
      setLastTranscriptionModel(desiredLocalModel as TranscriptModel);

      try {
        if (file.size > MAX_LOCAL_DECODE_FILE_BYTES) {
          throw new Error(copy.errors.localTooLarge);
        }

        const detectedDuration = await getAudioDurationSeconds(file);
        if (requestId !== activeRequestIdRef.current) return;
        if (detectedDuration !== null && detectedDuration > MAX_LOCAL_AUDIO_SECONDS) {
          throw new Error(copy.errors.localTooLong);
        }

        const audioData = await decodeAudioFile(file);
        if (requestId !== activeRequestIdRef.current) return;
        const decodedDuration = audioData.length / TARGET_SAMPLE_RATE;
        if (decodedDuration > MAX_LOCAL_AUDIO_SECONDS) {
          throw new Error(copy.errors.localTooLong);
        }
        setAudioDurationSeconds(Math.max(1, Math.round(decodedDuration)));

        const request: WorkerRequest = {
          type: "transcribe",
          requestId,
          audio: audioData,
          language: selectedLanguage ?? "english",
          model: desiredLocalModel,
        };
        workerRef.current?.postMessage(request, [audioData.buffer]);
      } catch (decodeError) {
        if (requestId !== activeRequestIdRef.current) return;
        setStatus("error");
        const decodeMessage = decodeError instanceof Error ? decodeError.message : "";
        const localizedDecodeErrors = new Set([
          copy.errors.localTooLarge,
          copy.errors.localTooLong,
        ]);
        setError(
          locale === "en" || localizedDecodeErrors.has(decodeMessage)
            ? decodeMessage || copy.errors.decodeFailed
            : copy.errors.decodeFailed,
        );
      }
    },
    [cancelTranscription, clearProgressState, copy, desiredLocalModel, isMobile, locale, preferFetchBlobUpload, selectedLanguage, status],
  );

  const handleFileSelected = useCallback(
    (file: File) => {
      if (!selectedLanguage) return;
      startTranscription(file);
    },
    [selectedLanguage, startTranscription],
  );

  const plainTextExport = useMemo(() => {
    if (segments.length > 0) {
      return segments
        .map((segment) => segment.text.trim())
        .filter(Boolean)
        .join(" ");
    }
    return output.trim();
  }, [output, segments]);

  const timestampedExport = useMemo(() => {
    if (segments.length > 0) {
      return segments
        .map((segment) => `[${formatSegmentTimestamp(segment.start)}] ${segment.text}`)
        .join("\n");
    }
    return output.trim();
  }, [output, segments]);

  const hasExportContent = useMemo(
    () => plainTextExport.trim().length > 0 || timestampedExport.trim().length > 0,
    [plainTextExport, timestampedExport],
  );

  const buildJsonExportPayload = useCallback((): TranscriptExportJson => {
    return {
      version: 1,
      createdAt: new Date().toISOString(),
      fileName: activeFileName ?? null,
      model: lastTranscriptionModel,
      language: selectedLanguage,
      text: output,
      segments,
    };
  }, [activeFileName, lastTranscriptionModel, output, segments, selectedLanguage]);

  const triggerDownload = useCallback((content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }, []);

  const writeToClipboard = useCallback(
    async (value: string, successMessage: string): Promise<boolean> => {
      if (!value.trim()) return false;
      let success = false;

      try {
        if (!navigator.clipboard) {
          throw new Error(copy.errors.clipboardUnavailable);
        }

        await navigator.clipboard.writeText(value);
        setCopyState("success");
        setCopyFeedback(successMessage);
        success = true;
      } catch (copyError) {
        setCopyState("error");
        setCopyFeedback(copy.errors.copyFailed);
        setError(
          locale === "en" && copyError instanceof Error
            ? copyError.message
            : copy.errors.copyFailed,
        );
      }

      if (copyResetTimeoutRef.current !== null) {
        window.clearTimeout(copyResetTimeoutRef.current);
      }

      copyResetTimeoutRef.current = window.setTimeout(() => {
        setCopyState("idle");
        setCopyFeedback(null);
      }, 1500);
      return success;
    },
    [copy, locale],
  );

  const exportAsWord = useCallback(
    async (title: string) => {
      const { Document, Packer, Paragraph, TextRun, AlignmentType } = await import("docx");
      const { saveAs } = await import("file-saver");

      const children: InstanceType<typeof Paragraph>[] = [];

      if (title.trim()) {
        children.push(
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 420 },
            children: [
              new TextRun({
                text: title.trim(),
                font: "Calibri Light",
                size: 44, // half-points: 44 = 22pt
                color: "FF0000",
                bold: true,
              }),
            ],
          }),
        );
      }

      children.push(
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { line: 360 },
          children: [
            new TextRun({
              text: plainTextExport,
              font: "Calibri",
              size: 24, // half-points: 24 = 12pt
            }),
          ],
        }),
      );

      const doc = new Document({
        sections: [{ children }],
      });

      const blob = await Packer.toBlob(doc);
      saveAs(blob, `transcription-${timestampForFilename(new Date())}.docx`);
    },
    [plainTextExport],
  );

  const exportAsPdf = useCallback(
    async (title: string) => {
      const { exportTranscriptAsPdf } = await import("@/lib/pdf-export");
      exportTranscriptAsPdf({
        fileName: `transcription-${timestampForFilename(new Date())}.pdf`,
        locale,
        text: plainTextExport,
        title,
      });
    },
    [locale, plainTextExport],
  );

  const handleSmartExport = useCallback(
    async (action: SmartExportAction): Promise<boolean> => {
      if (!hasExportContent) return false;

      if (action === "copy_text_only") {
        return writeToClipboard(plainTextExport, copy.copyTextSuccess);
      }

      if (action === "copy_with_timestamps") {
        return writeToClipboard(timestampedExport, copy.copyTimestampSuccess);
      }

      if (action === "export_json") {
        triggerDownload(
          JSON.stringify(buildJsonExportPayload(), null, 2),
          `transcription-${timestampForFilename(new Date())}.json`,
          "application/json;charset=utf-8",
        );
        setCopyState("success");
        setCopyFeedback(copy.jsonDownloaded);
        if (copyResetTimeoutRef.current !== null) {
          window.clearTimeout(copyResetTimeoutRef.current);
        }
        copyResetTimeoutRef.current = window.setTimeout(() => {
          setCopyState("idle");
          setCopyFeedback(null);
        }, 1500);
        return true;
      }

      if (action === "export_word" || action === "export_pdf") {
        setPendingExportAction(action);
        setExportTitle("");
        setIsExportTitleOpen(true);
        return true;
      }

      return false;
    },
    [
      buildJsonExportPayload,
      copy,
      hasExportContent,
      plainTextExport,
      timestampedExport,
      triggerDownload,
      writeToClipboard,
    ],
  );

  const confirmExportWithTitle = useCallback(async () => {
    setIsExportTitleOpen(false);
    if (!pendingExportAction) return;

    try {
      if (pendingExportAction === "export_word") {
        await exportAsWord(exportTitle);
      } else if (pendingExportAction === "export_pdf") {
        await exportAsPdf(exportTitle);
      }
    } catch (exportError) {
      console.error("Document export failed:", exportError);
      setError(copy.errors.exportFailed);
    } finally {
      setPendingExportAction(null);
    }
  }, [copy, exportAsPdf, exportAsWord, exportTitle, pendingExportAction]);

  const closeExportDialog = useCallback(() => {
    setIsExportTitleOpen(false);
    setPendingExportAction(null);
  }, []);

  useEffect(() => {
    if (!isExportTitleOpen) return;

    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    exportTitleInputRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeExportDialog();
        return;
      }

      if (event.key !== "Tab" || !exportDialogRef.current) return;
      const focusable = Array.from(
        exportDialogRef.current.querySelectorAll<HTMLElement>(
          'input:not([disabled]), button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [closeExportDialog, isExportTitleOpen]);

  const continueWithAi = useCallback(
    (url: string) => {
      window.open(url, "_blank", "noopener,noreferrer");
      void writeToClipboard(output.trim(), copy.copyTextSuccess);
    },
    [copy, output, writeToClipboard],
  );

  const totalAudioSeconds = useMemo(() => {
    if (audioDurationSeconds !== null) {
      return audioDurationSeconds;
    }

    if (totalChunks === null || totalChunks <= 0) {
      return null;
    }

    if (isMobile) {
      return totalChunks * CLOUD_CHUNK_DURATION_S;
    }

    return Math.round((totalChunks - 1) * LOCAL_AUDIO_STEP_S + LOCAL_CHUNK_LENGTH_S);
  }, [audioDurationSeconds, isMobile, totalChunks]);

  const progressLabel = useMemo(() => {
    if (progressPhase === "download") {
      if (downloadedBytes !== null && totalBytes !== null && totalBytes > 0) {
        const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
        const dlMB = (downloadedBytes / (1024 * 1024)).toFixed(1).padStart(totalMB.length, "\u00A0");
        const pct = progress.toFixed(0).padStart(3, "\u00A0");
        return copy.downloadingModel(dlMB, totalMB, pct);
      }
      const pct = progress.toFixed(0).padStart(3, "\u00A0");
      return copy.downloadingModelPercent(pct);
    }
    if (progressPhase === "transcribing") {
      const processedAudioSec =
        processedChunks !== null
          ? Math.round(
            processedChunks * (isMobile ? CLOUD_CHUNK_DURATION_S : LOCAL_AUDIO_STEP_S),
          )
          : null;
      const clampedProcessedAudioSec =
        processedAudioSec !== null && totalAudioSeconds !== null
          ? Math.min(processedAudioSec, totalAudioSeconds)
          : processedAudioSec;

      const fmtMin = (s: number) => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return copy.minuteValue(m, sec);
      };

      const timeStr =
        clampedProcessedAudioSec !== null && totalAudioSeconds !== null
          ? copy.transcribedRange(fmtMin(clampedProcessedAudioSec), fmtMin(totalAudioSeconds))
          : null;

      const sliceStr =
        totalSlices !== null && totalSlices > 1 && currentSlice !== null
          ? copy.slice(currentSlice, totalSlices)
          : null;

      const pctStr = `${progress.toFixed(0)}%`;

      return [sliceStr, timeStr, pctStr].filter(Boolean).join("  ·  ");
    }
    return "";
  }, [copy, currentSlice, downloadedBytes, isMobile, processedChunks, progress, progressPhase, totalAudioSeconds, totalBytes, totalSlices]);

  const roughAudioMinutes =
    totalAudioSeconds !== null
      ? Math.round(totalAudioSeconds / 60)
      : null;

  const etaLabel = useMemo(() => {
    if (progressPhase !== "transcribing") return null;

    // Before the first chunk completes we have no timing data —
    // show a rough estimate from audio duration instead of "calculating..."
    if (etaSeconds === null) {
      if (roughAudioMinutes !== null && roughAudioMinutes > 0) {
        const low = isMobile
          ? Math.max(1, Math.round(roughAudioMinutes / 15))
          : Math.max(1, Math.round(roughAudioMinutes / 5));
        const high = isMobile
          ? Math.max(2, Math.round(roughAudioMinutes / 6))
          : Math.max(2, Math.round(roughAudioMinutes / 2));
        return copy.roughEstimate(roughAudioMinutes, low, high);
      }
      return copy.etaCalculating;
    }
    if (etaSeconds <= 0) return copy.etaFinishing;
    return copy.etaRemaining(formatSegmentTimestamp(etaSeconds));
  }, [copy, etaSeconds, isMobile, progressPhase, roughAudioMinutes]);

  const busy =
    status === "loading" || status === "decoding" || status === "transcribing" || isCancelling;
  // Only show the compact file row (instead of dropzone) when there's an active file being processed.
  // During model preloading (status === "loading", activeFileName === null) we keep the dropzone visible.
  const uploadBusy = busy && activeFileName !== null;
  /** True once the model pipeline is loaded and ready to transcribe. */
  // Suppress the loading UI on page-refresh if the model was previously cached.
  // Only show the loading UI again if bytes are actually being transferred
  // (i.e. the browser cache was cleared and a real re-download is happening).
  const modelReady =
    status === "ready" ||
    status === "transcribing" ||
    status === "decoding" ||
    isMobile ||
    (wasModelEverLoaded && status === "loading" && !sawActiveDownload);
  const isCompiling = status === "loading" && loadingDetail === "compiling";
  /** True while local (desktop) transcription warms up before first chunk callback. */
  const isWarmingUp =
    !isMobile &&
    !isViaCloud &&
    status === "transcribing" &&
    processedChunks === 0 &&
    totalChunks !== null &&
    totalChunks > 0;

  // Live elapsed-seconds counter while local processing handles the first chunk.
  // This is the ONLY visual proof of activity during an otherwise silent 30-90 s wait.
  useEffect(() => {
    if (!isWarmingUp) return;
    const id = window.setInterval(() => setWarmUpElapsed((prev) => prev + 1), 1_000);
    return () => window.clearInterval(id);
  }, [isWarmingUp]);
  const showProgressBar = progressPhase === "transcribing" && !isWarmingUp;
  const showSkeleton =
    !output && activeFileName !== null && (status === "decoding" || status === "transcribing");
  const structuredDataJson = useMemo(
    () =>
      JSON.stringify(
        isTurkishPage
          ? [FAQ_JSON_LD_TR, HOW_TO_JSON_LD_TR]
          : [FAQ_JSON_LD_EN, HOW_TO_JSON_LD_EN],
      ),
    [isTurkishPage],
  );

  return (
    <main className="relative min-h-screen select-none overflow-y-auto px-4 py-4 sm:px-6 sm:py-5 lg:h-screen lg:overflow-hidden lg:px-8">
      <div className="mx-auto flex h-full w-full max-w-[1440px] flex-col">

        <header className="mb-6 flex items-start justify-between gap-6 pb-2">
          <div className="flex items-center">
            <div className="flex min-w-0 items-start gap-3.5 sm:items-center sm:gap-5">
              <div className="flex h-11 w-16 shrink-0 items-center text-emerald-200/90 sm:h-12 sm:w-[4.5rem]">
                <svg viewBox="0 0 72 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" className="h-10 w-full">
                  <rect x="1" y="15" width="3" height="10" rx="1.5" fill="currentColor" />
                  <rect x="7" y="9" width="3" height="22" rx="1.5" fill="currentColor" />
                  <rect x="13" y="4" width="3" height="32" rx="1.5" fill="currentColor" />
                  <rect x="19" y="11" width="3" height="18" rx="1.5" fill="currentColor" />
                  <rect x="25" y="7" width="3" height="26" rx="1.5" fill="currentColor" />
                  <path d="M34 11H69M34 20H62M34 29H54" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" opacity="0.58" />
                </svg>
              </div>
              <div className="min-w-0">
                <h1 className="font-display text-xl font-semibold text-white sm:text-[1.75rem]">
                  {copy.appTitle}
                </h1>
                <p className="mt-0.5 max-w-2xl text-sm leading-5 text-neutral-400">
                  {copy.appSubtitle}
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="font-semibold text-emerald-100/75">{copy.freeBadge}</span>
                  <span aria-hidden="true" className="text-neutral-700">/</span>
                  <span className="font-medium text-neutral-400">{copy.noSignup}</span>
                </div>
              </div>
            </div>
          </div>
          <nav
            aria-label={isTurkishPage ? "Arayüz dili" : "Interface language"}
            className="flex shrink-0 items-center gap-2.5 pt-1 text-xs"
          >
            {isTurkishPage ? (
              <Link href="/" hrefLang="en" className="font-medium text-neutral-400 transition-colors hover:text-neutral-100">EN</Link>
            ) : (
              <span aria-current="page" className="font-semibold text-neutral-100">EN</span>
            )}
            <span aria-hidden="true" className="text-neutral-700">/</span>
            {isTurkishPage ? (
              <span aria-current="page" className="font-semibold text-neutral-100">TR</span>
            ) : (
              <Link href="/tr" hrefLang="tr" className="font-medium text-neutral-400 transition-colors hover:text-neutral-100">TR</Link>
            )}
          </nav>
        </header>

        {/* ═══ Two-column layout: left = controls, right = transcript ═══ */}
        <div className="grid min-h-0 flex-1 gap-5 lg:grid-cols-[minmax(300px,0.82fr)_minmax(0,1.55fr)] xl:grid-cols-[minmax(330px,0.82fr)_minmax(0,1.65fr)]">

          {/* ── Left column: language, model status, upload, progress ── */}
          <div className="space-y-5 lg:min-h-0 lg:overflow-y-auto lg:pr-1">

        <div>
          <label htmlFor="audio-language" className="mb-2 flex items-center gap-2 text-xs font-semibold text-neutral-300">
            <span className="w-5 shrink-0 text-center font-display text-[10px] tabular-nums text-emerald-200/50">01</span>
            <span>{copy.stepLanguage}</span>
          </label>
          <div
            className={["relative block", isLangShaking ? "lang-shake" : ""].join(" ")}
            onAnimationEnd={() => setIsLangShaking(false)}
          >
            <LanguageSelect
              id="audio-language"
              ref={languageSelectRef}
              value={selectedLanguage}
              placeholder={copy.selectLanguage}
              options={languageOptions}
              onValueChange={(nextLanguage) => {
                setSelectedLanguage(nextLanguage as WhisperLanguage);
              }}
            />
          </div>
        </div>

        {/* Inline model status — only shows when model is actively loading/compiling */}
        {!modelReady && selectedLanguage && !isMobile && (
          <div role="status" aria-live="polite" className="mb-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-neutral-400">
                <svg className="size-3.5 animate-spin text-cyan-400" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>
                  {isCompiling
                    ? copy.compilingEngine
                    : progressPhase === "download"
                      ? (() => {
                          const pct = Math.round(progress);
                          const dl = typeof downloadedBytes === "number" ? (downloadedBytes / (1024 * 1024)).toFixed(0) : null;
                          const tot = typeof totalBytes === "number" && totalBytes > 0 ? (totalBytes / (1024 * 1024)).toFixed(0) : null;
                          return dl && tot ? copy.downloadingEngine(dl, tot) : copy.preparingEnginePercent(pct);
                        })()
                      : copy.preparingEngine}
                </span>
              </div>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="text-xs text-neutral-400 transition-colors hover:text-red-300"
              >
                {isCancelling ? copy.cancelling : copy.cancel}
              </button>
            </div>
            {(progressPhase === "download" || isCompiling) && (
              <div
                role="progressbar"
                aria-label={copy.preparingEngine}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={isCompiling ? undefined : Math.round(progress)}
                className="mt-2 h-1 overflow-hidden rounded-[2px] bg-neutral-800"
              >
                {isCompiling ? (
                  <div className="activity-pulse h-full w-full rounded-[2px] bg-cyan-300/55" />
                ) : (
                  <div
                    className="h-full rounded-[2px] bg-cyan-400/70 transition-all duration-500 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 2 — Upload */}
        <div id="upload-section" className="scroll-mt-24">
          {uploadBusy ? (
            /* While processing: hide the full dropzone, show only the compact file row */
            <div>
              <p className="mb-2 text-xs font-medium text-neutral-400">
                <span className="inline-flex items-center gap-2 normal-case">
                  <span className="w-5 shrink-0 text-center font-display text-[10px] tabular-nums text-emerald-200/50">02</span>
                  <span className="text-xs font-semibold text-neutral-300">{copy.stepUpload}</span>
                </span>
              </p>
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-[#151a17] px-3.5 py-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] sm:gap-3">
                <div className="flex min-w-0 items-center gap-2.5">
                  <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor" className="shrink-0 text-neutral-400"><path d="M560-360v-240l80 80 56-56-160-160-160 160 56 56 80-80v240h48Zm-80 200q-83 0-141.5-58.5T280-360v-400h400v400q0 83-58.5 141.5T480-160Zm0-80q50 0 85-35t35-85v-320H360v320q0 50 35 85t85 35ZM200-80q-33 0-56.5-23.5T120-160v-520h80v520h520v80H200Zm280-440Z" /></svg>
                  <span className="truncate text-sm text-neutral-200">{activeFileName}</span>
                </div>
                <span className="shrink-0 text-xs text-neutral-400 sm:ml-auto">{copy.processing}</span>
              </div>
            </div>
          ) : (
            <div>
              <p className="mb-2 text-xs font-medium text-neutral-400">
                <span className="inline-flex items-center gap-2 normal-case">
                  <span className="w-5 shrink-0 text-center font-display text-[10px] tabular-nums text-emerald-200/50">02</span>
                  <span className="text-xs font-semibold text-neutral-300">{copy.stepUpload}</span>
                </span>
              </p>
              <UploadDropzone
                disabled={!selectedLanguage || !modelReady}
                disabledReason={
                  !selectedLanguage ? copy.uploadDisabledLanguage : copy.uploadDisabledModel
                }
                locale={locale}
                onDisabledActivate={() => {
                  if (!selectedLanguage) {
                    setIsLangShaking(true);
                    languageSelectRef.current?.focus();
                  }
                }}
                onFileSelected={handleFileSelected}
              />
            </div>
          )}
        </div>
        {isWarmingUp ? (
          <div className="mt-3 rounded-lg bg-[#121714] p-4">
            {/* Header row */}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
              <div className="space-y-0.5">
                <p className="text-sm font-semibold text-neutral-100">
                  {copy.transcriptionInProgress}
                </p>
                <p className="text-xs text-neutral-400">
                  {copy.warmupDescription}
                </p>
              </div>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="inline-flex min-h-9 items-center gap-1.5 self-start px-2 py-1 text-xs font-medium text-red-300 transition-colors hover:text-red-100 disabled:cursor-not-allowed disabled:opacity-60 sm:shrink-0"
              >
                <Square className="size-3.5" />
                {isCancelling ? copy.cancelling : copy.cancel}
              </button>
            </div>

            {/* Stats row */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 text-xs font-medium tabular-nums text-cyan-200/85">
                <Clock3 className="size-3.5" style={{ animationDuration: "3s" }} />
                {copy.elapsed(warmUpElapsed)}
              </span>

              {totalSlices !== null && totalSlices > 1 && currentSlice !== null ? (
                <span className="inline-flex items-center text-xs text-neutral-400">
                  {copy.slice(currentSlice, totalSlices)}
                </span>
              ) : null}

              {totalChunks !== null ? (
                <span className="inline-flex items-center text-xs text-neutral-400">
                  {copy.segmentCount(0, totalChunks)}
                </span>
              ) : null}

              {roughAudioMinutes !== null ? (
                <span className="inline-flex items-center text-xs text-neutral-400">
                  {copy.audioMinutes(roughAudioMinutes)}
                </span>
              ) : null}
            </div>

            {/* Activity bar */}
            <div role="progressbar" aria-label={copy.transcriptionInProgress} className="mt-3 h-1 overflow-hidden rounded-[2px] bg-neutral-800">
              <div className="activity-pulse h-full w-full rounded-[2px] bg-cyan-300/70" />
            </div>

            <p className="mt-2 text-xs text-neutral-400">
              {copy.keepTabActive}
            </p>
          </div>
        ) : null}

        {showProgressBar ? (
          <div className="mt-3 space-y-2.5 rounded-lg bg-[#121714] p-3">
            {/* ── Top row: label + cancel ─────────────────────────────────── */}
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs font-medium text-neutral-200 sm:text-sm" style={{ fontVariantNumeric: "tabular-nums" }}>{progressLabel}</p>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="inline-flex min-h-9 shrink-0 items-center gap-1.5 px-2 py-1 text-xs font-medium text-red-300 transition-colors hover:text-red-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Square className="size-3.5" />
                {isCancelling ? copy.cancelling : copy.cancel}
              </button>
            </div>

            {/* ── Progress bar ────────────────────────────────────────────── */}
            <div
              role="progressbar"
              aria-label={copy.transcriptionInProgress}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress)}
              className="h-2 overflow-hidden rounded-[2px] bg-neutral-900/90"
            >
              <div
                className="h-full rounded-[2px] bg-cyan-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* ── Slice pip track (only when there are multiple slices) ───── */}
            {totalSlices !== null && totalSlices > 1 ? (
              <div className="flex items-center gap-1">
                {Array.from({ length: totalSlices }).map((_, i) => (
                  <div
                    key={i}
                    className={[
                      "h-1 flex-1 rounded-[1px] transition-colors duration-300",
                      currentSlice !== null && i < currentSlice
                        ? "bg-cyan-400"
                        : currentSlice !== null && i === currentSlice - 1
                          ? "bg-cyan-400/60"
                          : "bg-neutral-700",
                    ].join(" ")}
                  />
                ))}
              </div>
            ) : null}

            {/* ── ETA row ─────────────────────────────────────────────────── */}
            {etaLabel ? (
              <p className="text-xs text-neutral-400" style={{ fontVariantNumeric: "tabular-nums" }}>{etaLabel}</p>
            ) : null}
            {loadingDetail && loadingDetail !== "compiling" ? (
              <p role="status" aria-live="polite" className="text-xs text-neutral-400">{loadingDetail}</p>
            ) : null}
          </div>
        ) : null}

        {status === "decoding" && !showProgressBar ? (
          <div role="status" aria-live="polite" className="mt-3 flex flex-col gap-2 rounded-lg bg-[#121714] p-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-neutral-300 sm:text-sm">{loadingDetail || copy.decodingAudio}</p>
            <button
              type="button"
              onClick={cancelTranscription}
              disabled={isCancelling}
              className="inline-flex min-h-9 items-center gap-1.5 px-2 py-1 text-xs font-medium text-red-300 transition-colors hover:text-red-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Square className="size-3.5" />
              {isCancelling ? copy.cancelling : copy.cancel}
            </button>
          </div>
        ) : null}

        {error ? (
          <div role="alert" className="mt-3 inline-flex items-center gap-2 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle className="size-4" />
            <span>{error}</span>
          </div>
        ) : null}

          </div>{/* end left column */}

          {/* ── Right column: transcript output ── */}
          <div className="flex min-h-0 flex-col">

        {!output ? (
          <div className="flex min-h-[320px] flex-col overflow-hidden rounded-lg bg-[#151a17] shadow-[inset_0_1px_0_rgba(255,255,255,0.045)] lg:min-h-0 lg:flex-1">
            <div className="flex h-11 shrink-0 items-center justify-between bg-[#19201b] px-4">
              <div className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-300/80" />
                <span className="font-display text-sm font-semibold text-neutral-200">
                  {copy.transcriptPanelTitle}
                </span>
              </div>
              <span className="text-[11px] font-medium text-neutral-400">
                {showSkeleton ? copy.transcriptionInProgress : copy.transcriptReady}
              </span>
            </div>

            <div className="flex flex-1 flex-col items-center justify-center px-6 py-10 text-center">
              <svg
                aria-hidden="true"
                className={[
                  "mb-8 h-auto w-full max-w-[36rem]",
                  showSkeleton ? "text-cyan-200/70" : "text-emerald-100/55",
                ].join(" ")}
                fill="none"
                viewBox="0 0 640 176"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M18 88h14l9-26 11 52 12-76 12 100 11-64 12 28 11-48 12 68 11-34h15l9-20 11 40 12-58 11 76 12-48 10 20h43"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="4"
                />
                <path d="M286 48v80" stroke="currentColor" strokeLinecap="round" strokeWidth="2" opacity="0.22" />
                <path d="M318 48h286M318 88h226M318 128h266" stroke="currentColor" strokeLinecap="round" strokeWidth="4" opacity="0.72" />
                <path d="M318 64h198M318 104h272M318 144h174" stroke="currentColor" strokeLinecap="round" strokeWidth="2" opacity="0.28" />
                {showSkeleton ? (
                  <path className="transcript-scan" d="M318 30v116" stroke="currentColor" strokeLinecap="round" strokeWidth="3" />
                ) : null}
              </svg>

              <p className="font-display text-base font-semibold text-neutral-200">
                {showSkeleton ? copy.transcribingPlaceholder : copy.rotatingPhrases[0]}
              </p>
              <p className="mt-1.5 max-w-md text-sm leading-5 text-neutral-400">
                {showSkeleton ? copy.textWillAppear : copy.transcriptWillAppear}
              </p>
            </div>
          </div>
        ) : (
        <>
        <div className={["overflow-hidden rounded-lg bg-[#151a17] shadow-[inset_0_1px_0_rgba(255,255,255,0.045)] lg:min-h-0 lg:flex-1 lg:overflow-y-auto", justCompleted ? "transcript-flash" : ""].join(" ")}>
          <div className="flex flex-wrap items-center justify-between gap-2 bg-[#19201b] px-4 py-2.5">
            <div className="flex min-w-0 flex-wrap items-center gap-2 sm:gap-3">
              <span className="inline-flex items-center gap-2 font-display text-sm font-semibold text-neutral-200">
                <span className="size-1.5 rounded-full bg-emerald-300/80" />
                {copy.transcriptPanelTitle}
              </span>
              {output ? (
                <span className="text-xs font-medium text-neutral-400">
                  {copy.wordCount(output.trim().split(/\s+/).filter(Boolean).length)}
                </span>
              ) : null}
              {segments.length > 0 ? (
                <div role="group" aria-label={copy.transcriptViewLabel} className="flex w-full items-center rounded-md border border-white/[0.08] bg-[#121513] p-0.5 sm:w-auto">
                  <button
                    type="button"
                    aria-pressed={viewMode === "plain"}
                    onClick={() => setViewMode("plain")}
                    className={[
                      "flex-1 rounded px-2 py-1 text-[11px] font-medium transition-colors sm:flex-none sm:px-2.5 sm:text-xs",
                      viewMode === "plain"
                        ? "bg-white/[0.07] text-neutral-100"
                        : "text-neutral-400 hover:text-neutral-200",
                    ].join(" ")}
                  >
                    {copy.plainText}
                  </button>
                  <button
                    type="button"
                    aria-pressed={viewMode === "timestamps"}
                    onClick={() => setViewMode("timestamps")}
                    className={[
                      "flex-1 rounded px-2 py-1 text-[11px] font-medium transition-colors sm:flex-none sm:px-2.5 sm:text-xs",
                      viewMode === "timestamps"
                        ? "bg-white/[0.07] text-neutral-100"
                        : "text-neutral-400 hover:text-neutral-200",
                    ].join(" ")}
                  >
                    {copy.withTimestamps}
                  </button>
                </div>
              ) : null}
            </div>
            <div className="flex w-full flex-wrap items-center gap-1.5 sm:w-auto">
              <button
                type="button"
                disabled={!hasExportContent}
                onClick={async () => {
                  if (copiedBtnTimeoutRef.current) window.clearTimeout(copiedBtnTimeoutRef.current);
                  const success = await handleSmartExport("copy_text_only");
                  if (success) {
                    setCopiedButton("text");
                    copiedBtnTimeoutRef.current = window.setTimeout(() => setCopiedButton(null), 1500);
                  }
                }}
                className={[
                  "inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                  copiedButton === "text"
                    ? "bg-emerald-400/10 text-emerald-200"
                    : "text-neutral-300 hover:bg-white/[0.05] hover:text-white",
                ].join(" ")}
              >
                {copiedButton === "text" ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                {copiedButton === "text" ? copy.copied : copy.copyText}
              </button>
              <button
                type="button"
                disabled={!hasExportContent}
                onClick={async () => {
                  if (copiedBtnTimeoutRef.current) window.clearTimeout(copiedBtnTimeoutRef.current);
                  const success = await handleSmartExport("copy_with_timestamps");
                  if (success) {
                    setCopiedButton("timestamps");
                    copiedBtnTimeoutRef.current = window.setTimeout(() => setCopiedButton(null), 1500);
                  }
                }}
                className={[
                  "inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                  copiedButton === "timestamps"
                    ? "bg-emerald-400/10 text-emerald-200"
                    : "text-neutral-300 hover:bg-white/[0.05] hover:text-white",
                ].join(" ")}
              >
                {copiedButton === "timestamps" ? <Check className="size-3.5" /> : <Clock3 className="size-3.5" />}
                {copiedButton === "timestamps" ? copy.copied : copy.timestamps}
              </button>
              <button
                type="button"
                disabled={!hasExportContent}
                onClick={() => void handleSmartExport("export_json")}
                className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:bg-white/[0.05] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                <FileText className="size-3.5" />
                {copy.exportJson}
              </button>
              <button
                type="button"
                disabled={!hasExportContent}
                onClick={() => void handleSmartExport("export_word")}
                className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:bg-white/[0.05] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                <svg className="size-4 shrink-0" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M28 2H14l-2 2v6H4L2 12v16l2 2h14l2-2v-6h8l2-2V4l-2-2z" fill="#fff"/><path d="M28 2H14l-2 2v6h10l2 2v12h4l2-2V4l-2-2z" fill="#2B579A"/><path d="M24 22V10H12v20h14l-2-8z" fill="#2B579A"/><path d="M2 12l2-2h8l2 2v16l-2 2H4l-2-2V12z" fill="#2B579A"/><path d="M4 12h10v16H4V12z" fill="#2B579A"/><path d="M18.5 14L17 24h-1.5l-1-6.5L13.5 24H12l-1.5-10h1.3l.9 6.7 1.1-6.7h1.4l1 6.7.9-6.7h1.4z" fill="#fff"/></svg>
                Word
              </button>
              <button
                type="button"
                disabled={!hasExportContent}
                onClick={() => void handleSmartExport("export_pdf")}
                className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:bg-white/[0.05] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                <svg className="size-4 shrink-0" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 2h16l8 8v20l-2 2H4L2 30V4l2-2z" fill="#fff"/><path d="M20 2v8h8" fill="#E5252A"/><path d="M4 2h16v8h8v20l-2 2H4L2 30V4l2-2z" fill="#E5252A"/><path d="M8.5 24V14h3.2c1 0 1.7.3 2.2.8.5.5.7 1.2.7 2s-.2 1.5-.7 2c-.5.5-1.2.8-2.2.8H10v4.4H8.5zm1.5-5.8h1.5c.6 0 1-.1 1.3-.4.3-.3.4-.7.4-1.2s-.1-.9-.4-1.2c-.3-.3-.7-.4-1.3-.4H10v3.2zm6.5 5.8V14h2.8c1.6 0 2.7.4 3.4 1.2.7.8 1 2 1 3.4 0 1.5-.4 2.6-1.1 3.4-.7.8-1.8 1.2-3.3 1.2h-2.8zm1.5-1.3h1.2c1 0 1.7-.3 2.2-.8.5-.5.7-1.4.7-2.5s-.2-1.9-.7-2.5c-.5-.5-1.2-.8-2.2-.8H18v6.6z" fill="#fff"/></svg>
                PDF
              </button>
            </div>
          </div>

          <div>
            {segments.length > 0 && viewMode === "timestamps" ? (
              <div className="max-h-[560px] select-text overflow-y-auto">
                <ul className="divide-y divide-white/[0.06]">
                  {segments.map((segment, index) => (
                    <li
                      key={`${segment.start}-${segment.end}-${index}`}
                      className="grid grid-cols-[64px_1fr] gap-2.5 px-5 py-3 sm:grid-cols-[80px_1fr] sm:gap-3"
                    >
                      <span className="pt-0.5 text-xs text-neutral-400">
                        [{formatSegmentTimestamp(segment.start)}]
                      </span>
                      <p
                        className="text-sm leading-6 text-neutral-200 font-sans"
                      >
                        {segment.text}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            ) : output ? (
              <textarea
                ref={outputTextareaRef}
                readOnly
                aria-label={copy.transcriptLabel}
                value={output}
                className={[
                  "h-[360px] w-full resize-none select-text overflow-y-auto border-0 bg-transparent p-5 font-sans text-sm leading-6 text-neutral-200 outline-none",
                ].join(" ")}
              />
            ) : null}
          </div>
        </div>
        <p role="status" aria-live="polite" className={copyState === "error" ? "sr-only text-red-300" : "sr-only"}>
          {copyFeedback}
        </p>
        {justCompleted ? (
          <p role="status" aria-live="polite" className="sr-only">{copy.transcriptionComplete}</p>
        ) : null}

        {output.trim() && status !== "transcribing" && status !== "decoding" && status !== "loading" ? (
          <div className="mt-3 bg-[#121714] px-4 py-3.5">
            <p className="mb-0.5 font-display text-sm font-semibold text-neutral-200">{copy.continueWithAi}</p>
            <p className="mb-3 text-xs leading-relaxed text-neutral-400">
              {copy.continueWithAiDescription}
            </p>
            <div className="flex flex-wrap gap-2">
              <button type="button" aria-label={copy.opensInNewTab("ChatGPT")} onClick={() => continueWithAi("https://chatgpt.com/")} className="inline-flex min-h-8 items-center gap-1.5 px-1.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:text-white">
                <svg height="14" width="14" viewBox="0 0 256 260" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.062 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.6zM37.158 197.93a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803zM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405zm179.466 41.695l-63.08-36.63L161.73 77.2a.82.82 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.243zm21.742-32.69l-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391zM88.061 139.098l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697zm11.868-25.58l28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434z" fill="#10A37F"/></svg>
                ChatGPT
              </button>
              <button type="button" aria-label={copy.opensInNewTab("Claude")} onClick={() => continueWithAi("https://claude.ai/")} className="inline-flex min-h-8 items-center gap-1.5 px-1.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:text-white">
                <svg height="14" width="14" viewBox="0 0 256 257" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M50.228 170.321l50.357-28.257.843-2.463-.843-1.361h-2.463l-8.425-.519-28.776-.777-24.951-1.037-24.175-1.296L5.703 133.314 0 125.796l.583-3.759 5.12-3.435 7.324.648 16.202 1.102 24.304 1.685 17.628 1.037 26.119 2.722h4.148l.583-1.685-1.426-1.037-1.102-1.037-25.246-17.08-27.22-18.017-14.258-10.37-7.713-5.25-3.888-4.925-1.686-10.758 7-7.713 9.398.648 2.398.648 9.527 7.324 20.35 15.748 26.572 19.573 3.889 3.24 1.555-1.102.194-.777-1.749-2.916-14.453-26.119-25.989-26.572-6.87-11.018-1.814-6.61c-.648-2.722-1.102-4.99-1.102-7.777l7.971-10.823 4.407-1.426 10.63 1.426 4.471 3.889 6.611 15.1 10.694 23.786 16.591 32.34 4.861 9.592 2.851 8.879.972 2.722h1.685v-1.555l1.361-18.212 2.527-22.36 2.463-28.775.843-8.102 4.018-9.721 7.971-5.25 6.222 2.981 5.12 7.324-.713 4.731-3.046 19.767-5.962 30.979-3.889 20.736 2.269 0 2.592-2.592 10.5-13.934 17.628-22.035 7.778-8.749 9.073-9.657 5.833-4.602 11.018 0 8.101 12.054-3.629 12.444-11.342 14.387-9.398 12.185-13.48 18.147-8.425 14.517 1.296 1.167 2.009-.195 30.461-6.481 16.462-2.981 19.637-3.37 8.879 4.148.972 4.212-3.5 8.62-22.299 5.185-24.627 4.925-36.682 8.685-.454.324.519.648 16.526 1.555 7.065.389 17.304 0 32.211 2.398 8.425 5.574 5.055 6.804-.843 5.185-12.962 6.61-17.499-4.148-40.83-9.721-13.999-3.5-1.944 0v1.167l11.666 11.406 21.387 19.314 26.767 24.887 1.361 6.157-3.435 4.861-.629-.519-23.526-17.693-9.073-7.971-20.545-17.306-1.361 0v1.815l4.731 6.934 25.017 37.59 1.296 11.536-1.815 3.759-6.481 2.269-7.13-1.296-14.647-20.545-15.101-23.137-12.184-20.739-1.491.843-7.194 77.448-3.37 3.953-7.777 2.981-6.481-4.925-3.435-7.972 3.435-15.748 4.148-20.111 3.37-16.328 3.046-20.286 1.815-6.74-.13-.454-1.491.195-15.296 21 23.267 31.432 18.406 19.703 3.593 1.749-7.648-3.953.713-7.065 4.277-6.287 25.471-32.405 15.36-20.091 9.916-11.601-.065-1.685-.583 0L44.071 198.125l-12.054 1.555-5.185-4.861.648-7.971 2.463-2.593 20.35-13.999-.064-.065z" fill="#D97757"/></svg>
                Claude
              </button>
              <button type="button" aria-label={copy.opensInNewTab("Gemini")} onClick={() => continueWithAi("https://gemini.google.com/")} className="inline-flex min-h-8 items-center gap-1.5 px-1.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:text-white">
                <svg height="14" width="14" viewBox="0 0 73 73" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M36.5 72.76c-2.81-18.787-17.473-33.45-36.26-36.26C18.787 33.69 33.45 19.027 36.26.24c2.81 18.787 17.473 33.45 36.26 36.26-18.787 2.81-33.45 17.473-36.26 36.26z" fill="#076EFF"/></svg>
                Gemini
              </button>
              <button type="button" aria-label={copy.opensInNewTab("Grok")} onClick={() => continueWithAi("https://grok.com/")} className="inline-flex min-h-8 items-center gap-1.5 px-1.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:text-white">
                <svg height="14" width="14" viewBox="0 0 256 246" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M63.831 56.843C91.3 29.363 131.465 21.978 165.542 34.973l2.314.917c7.645 2.844 14.309 6.89 19.507 10.652l-28.857 13.341c-26.869-11.285-57.649-3.608-76.435 15.2-25.406 25.414-30.54 69.484-.765 97.96L0 245.765c4.296-5.924 9.457-11.574 14.749-17.178l5.816-6.13 2.608-2.775c15.531-16.654 28.81-33.769 20.496-56.708l-.766-1.98C28.311 125.496 36.809 83.897 63.83 56.843zM220.786 35.256L256 0l-10.128 14.069C224.778 43.785 215.416 62.493 224.762 102.728l-.065-.065c7.23 30.728-.504 64.803-25.472 89.802-31.478 31.538-81.852 38.558-123.336 10.17l28.923-13.407c26.476 10.41 55.442 5.839 76.26-15.003 20.818-20.843 25.493-51.2 15.03-76.461-1.989-4.79-7.952-5.993-12.125-2.91L85.87 157.755 220.786 35.147v.109z" fill="#fff"/></svg>
                Grok
              </button>
            </div>
          </div>
        ) : null}
        </>
        )}{/* end placeholder/transcript conditional */}

          </div>{/* end right column */}
        </div>{/* end grid */}

        {/* ── Title input modal for Word/PDF export ── */}
        {isExportTitleOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
            <div
              ref={exportDialogRef}
              role="dialog"
              aria-modal="true"
              aria-labelledby="export-dialog-title"
              aria-describedby="export-dialog-description"
              className="w-full max-w-md rounded-lg bg-[#1a1e1b] p-6 shadow-[0_12px_28px_rgba(0,0,0,0.36)]"
            >
              <h2 id="export-dialog-title" className="mb-1 text-base font-semibold text-neutral-100">
                {copy.documentTitle}
              </h2>
              <p id="export-dialog-description" className="mb-4 text-xs text-neutral-400">
                {copy.documentTitleDescription}
              </p>
              <label htmlFor="export-document-title" className="sr-only">
                {copy.documentTitleInputLabel}
              </label>
              <input
                id="export-document-title"
                ref={exportTitleInputRef}
                type="text"
                value={exportTitle}
                onChange={(event) => setExportTitle(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void confirmExportWithTitle();
                }}
                placeholder={copy.documentTitlePlaceholder}
                className="mb-4 w-full rounded-lg border border-white/10 bg-neutral-800/60 px-4 py-2.5 text-sm text-neutral-100 outline-none placeholder:text-neutral-400 focus:border-emerald-200/40"
              />
              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={closeExportDialog}
                  className="rounded-lg px-4 py-2 text-sm text-neutral-400 transition-colors hover:text-neutral-200"
                >
                  {copy.cancel}
                </button>
                <button
                  type="button"
                  onClick={() => void confirmExportWithTitle()}
                  className="rounded-md bg-emerald-800 px-4 py-2 text-sm font-medium text-emerald-50 transition-colors hover:bg-emerald-700"
                >
                  {pendingExportAction === "export_word" ? copy.downloadWord : copy.downloadPdf}
                </button>
              </div>
            </div>
          </div>
        )}

        <footer className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 pb-1 text-xs">
          <p className="text-neutral-400">
            {copy.developedBy}{" "}
            <span className="font-medium text-neutral-300">Onat Özmen</span>
          </p>
          <div className="flex flex-wrap items-center gap-5">
            <a
              href={isTurkishPage ? "/tr/privacy" : "/privacy"}
              className="text-neutral-400 transition-colors hover:text-neutral-200"
            >
              {copy.privacyPolicy}
            </a>
            <a
              href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250"
              target="_blank"
              rel="noopener noreferrer"
              className="text-neutral-400 transition-colors hover:text-neutral-200"
            >
              LinkedIn <span aria-hidden="true">↗</span>
            </a>
          </div>
        </footer>

      </div>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: structuredDataJson,
        }}
      />
    </main>
  );
}
