export type AppLocale = "en" | "tr";

export type AppCopy = {
  languageNames: Record<string, string>;
  freeBadge: string;
  noSignup: string;
  appTitle: string;
  appSubtitle: string;
  stepLanguage: string;
  selectLanguage: string;
  stepUpload: string;
  processing: string;
  cancel: string;
  cancelling: string;
  compilingEngine: string;
  downloadingEngine: (downloaded: string, total: string) => string;
  preparingEnginePercent: (percent: number) => string;
  preparingEngine: string;
  transcriptionInProgress: string;
  warmupDescription: string;
  elapsed: (seconds: number) => string;
  slice: (current: number, total: number) => string;
  segmentCount: (processed: number, total: number) => string;
  audioMinutes: (minutes: number) => string;
  keepTabActive: string;
  decodingAudio: string;
  downloadingModel: (downloaded: string, total: string, percent: string) => string;
  downloadingModelPercent: (percent: string) => string;
  minuteValue: (minutes: number, seconds: number) => string;
  transcribedRange: (processed: string, total: string) => string;
  roughEstimate: (minutes: number, low: number, high: number) => string;
  etaCalculating: string;
  etaFinishing: string;
  etaRemaining: (timestamp: string) => string;
  uploadToCloud: string;
  secureSessionRetry: (seconds: number) => string;
  secureSessionBusy: (seconds: number) => string;
  uploadRetry: (seconds: number) => string;
  cloudConnectionRetry: (seconds: number) => string;
  cloudRateLimitRetry: (seconds: number) => string;
  cloudBusyRetry: (seconds: number) => string;
  transcribingChunk: (current: number, total: number) => string;
  transcribingAudio: string;
  longRecording: string;
  preparingUpload: string;
  startingSecureSession: string;
  preparingChunk: (current: number, total: number) => string;
  preparingAudio: string;
  uploadingChunk: (current: number, total: number) => string;
  uploadingSecurely: string;
  transcribingPlaceholder: string;
  textWillAppear: string;
  rotatingPhrases: string[];
  transcriptWillAppear: string;
  transcriptPanelTitle: string;
  transcriptReady: string;
  wordCount: (count: number) => string;
  plainText: string;
  withTimestamps: string;
  copied: string;
  copyText: string;
  timestamps: string;
  exportJson: string;
  continueWithAi: string;
  continueWithAiDescription: string;
  opensInNewTab: (service: string) => string;
  developedBy: string;
  privacyPolicy: string;
  documentTitle: string;
  documentTitleDescription: string;
  documentTitlePlaceholder: string;
  documentTitleInputLabel: string;
  downloadWord: string;
  downloadPdf: string;
  transcriptLabel: string;
  transcriptViewLabel: string;
  copyTextSuccess: string;
  copyTimestampSuccess: string;
  jsonDownloaded: string;
  transcriptionComplete: string;
  uploadDisabledLanguage: string;
  uploadDisabledModel: string;
  dropzone: {
    unsupportedType: string;
    uploadLabel: string;
    dropHere: string;
    uploadPrompt: string;
    instantConversion: string;
    or: string;
    browse: string;
    acceptedFormats: string;
    selectedFile: (name: string) => string;
  };
  errors: {
    webAudioUnsupported: string;
    workersUnsupported: string;
    workerUnexpected: string;
    workerUnavailable: string;
    sessionStartFailed: string;
    cloudUploadFailed: string;
    cloudTranscriptionFailed: string;
    cloudRequestFailed: (status: number) => string;
    mobileTooLong: string;
    mobileTooLarge: string;
    localTooLong: string;
    localTooLarge: string;
    sessionGrantMissing: string;
    decodeFailed: string;
    clipboardUnavailable: string;
    copyFailed: string;
    exportFailed: string;
  };
};

const EN_COPY: AppCopy = {
  languageNames: {
    english: "English",
    turkish: "Turkish",
    spanish: "Spanish",
    french: "French",
    german: "German",
    italian: "Italian",
    portuguese: "Portuguese",
    russian: "Russian",
    arabic: "Arabic",
    hindi: "Hindi",
    japanese: "Japanese",
    korean: "Korean",
  },
  freeBadge: "100% Free",
  noSignup: "No sign-up required",
  appTitle: "Audio Transcription",
  appSubtitle: "Convert audio to text instantly. Supports MP3, WAV, M4A and more.",
  stepLanguage: "Audio language",
  selectLanguage: "Select audio language",
  stepUpload: "Audio file",
  processing: "Processing...",
  cancel: "Cancel",
  cancelling: "Cancelling...",
  compilingEngine: "Compiling GPU shaders - first time only...",
  downloadingEngine: (downloaded, total) => `Downloading AI engine - ${downloaded} / ${total} MB`,
  preparingEnginePercent: (percent) => `Preparing AI engine - ${percent}%`,
  preparingEngine: "Preparing AI engine...",
  transcriptionInProgress: "Transcription in progress",
  warmupDescription: "Initial segment processing - the local engine is warming up. This takes 30-90 seconds the first time.",
  elapsed: (seconds) => `${seconds}s elapsed`,
  slice: (current, total) => `Slice ${current} / ${total}`,
  segmentCount: (processed, total) => `${processed} / ${total} segments`,
  audioMinutes: (minutes) => `~${minutes} min audio`,
  keepTabActive: "Keep this tab active while processing.",
  decodingAudio: "Decoding audio...",
  downloadingModel: (downloaded, total, percent) => `Downloading model... ${downloaded} / ${total} MB (${percent}%)`,
  downloadingModelPercent: (percent) => `Downloading model... ${percent}%`,
  minuteValue: (minutes, seconds) => seconds === 0 ? `${minutes} min` : `${minutes}:${String(seconds).padStart(2, "0")} min`,
  transcribedRange: (processed, total) => `${processed} / ${total} transcribed`,
  roughEstimate: (minutes, low, high) => `Audio length ~${minutes} min - estimated processing time: ${low}-${high} min`,
  etaCalculating: "Estimated time: calculating...",
  etaFinishing: "Estimated time: finishing...",
  etaRemaining: (timestamp) => `Estimated remaining: ${timestamp}`,
  uploadToCloud: "Uploading to cloud...",
  secureSessionRetry: (seconds) => `Secure session setup failed. Retrying in ${seconds}s...`,
  secureSessionBusy: (seconds) => `Secure session is busy. Retrying in ${seconds}s...`,
  uploadRetry: (seconds) => `Upload stalled. Retrying in ${seconds}s...`,
  cloudConnectionRetry: (seconds) => `Cloud connection issue. Retrying in ${seconds}s...`,
  cloudRateLimitRetry: (seconds) => `Cloud rate limit reached. Retrying in ${seconds}s...`,
  cloudBusyRetry: (seconds) => `Cloud is busy. Retrying in ${seconds}s...`,
  transcribingChunk: (current, total) => `Transcribing chunk ${current} of ${total}...`,
  transcribingAudio: "Transcribing audio...",
  longRecording: "Long recording detected. Splitting into mobile-safe chunks...",
  preparingUpload: "Preparing audio for upload...",
  startingSecureSession: "Starting secure cloud session...",
  preparingChunk: (current, total) => `Preparing chunk ${current} of ${total}...`,
  preparingAudio: "Preparing audio...",
  uploadingChunk: (current, total) => `Uploading chunk ${current} of ${total} securely...`,
  uploadingSecurely: "Uploading audio securely...",
  transcribingPlaceholder: "Transcribing your audio...",
  textWillAppear: "Text will start appearing here shortly",
  rotatingPhrases: [
    "Drop an audio file to get your transcript",
    "Supports MP3, WAV, M4A and 6 more formats",
    "Desktop transcription runs 100% locally",
    "Powered by Whisper and Moonshine AI",
    "Get timestamped output for subtitles",
    "No account needed - completely free",
    "Works with lectures, meetings and podcasts",
    "Export as text, timestamps, JSON, Word or PDF",
    "Transcribe in 12 different languages",
    "Your audio stays private on desktop",
  ],
  transcriptWillAppear: "Transcript will appear here",
  transcriptPanelTitle: "Transcript",
  transcriptReady: "Ready for audio",
  wordCount: (count) => `${count} ${count === 1 ? "word" : "words"}`,
  plainText: "Plain text",
  withTimestamps: "With timestamps",
  copied: "Copied",
  copyText: "Copy text",
  timestamps: "Timestamps",
  exportJson: "JSON",
  continueWithAi: "Continue with AI",
  continueWithAiDescription: "Open your transcript in an AI chat. It is copied to your clipboard automatically.",
  opensInNewTab: (service) => `Open transcript in ${service} (opens in a new tab)`,
  developedBy: "Developed by",
  privacyPolicy: "Privacy Policy",
  documentTitle: "Document title",
  documentTitleDescription: "Add a title for your document, or leave it blank to skip.",
  documentTitlePlaceholder: "Enter title (optional)",
  documentTitleInputLabel: "Document title (optional)",
  downloadWord: "Download Word",
  downloadPdf: "Download PDF",
  transcriptLabel: "Transcription output",
  transcriptViewLabel: "Transcript display mode",
  copyTextSuccess: "Text copied",
  copyTimestampSuccess: "Timestamped text copied",
  jsonDownloaded: "JSON downloaded",
  transcriptionComplete: "Transcription complete",
  uploadDisabledLanguage: "Select the audio language before uploading a file.",
  uploadDisabledModel: "Wait for the local transcription engine to finish loading.",
  dropzone: {
    unsupportedType: "Unsupported file type. Upload MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM, or OPUS.",
    uploadLabel: "Upload an audio file",
    dropHere: "Drop your audio file here",
    uploadPrompt: "Upload audio file to transcribe (MP3, WAV, M4A)",
    instantConversion: "Convert audio to text instantly",
    or: "or",
    browse: "browse from your device",
    acceptedFormats: "Accepted formats: .mp3, .wav, .m4a, .mp4, .ogg, .flac, .aac, .webm, .opus",
    selectedFile: (name) => `Selected file: ${name}`,
  },
  errors: {
    webAudioUnsupported: "Web Audio API is not supported in this browser.",
    workersUnsupported: "Web Workers are not supported in this browser.",
    workerUnexpected: "The transcription worker encountered an unexpected error.",
    workerUnavailable: "The transcription worker is not available.",
    sessionStartFailed: "Could not start the secure transcription session.",
    cloudUploadFailed: "Cloud upload failed after multiple retries.",
    cloudTranscriptionFailed: "Cloud transcription failed. Please try again.",
    cloudRequestFailed: (status) => `Cloud transcription failed (HTTP ${status}).`,
    mobileTooLong: "This recording is longer than 2 hours on mobile. Split it into parts or use desktop.",
    mobileTooLarge: "This file is too large for mobile browser processing. Use desktop or compress the recording.",
    localTooLong: "This recording is longer than 2 hours. Split it into smaller parts.",
    localTooLarge: "This file is too large for safe in-browser decoding. Compress it or split it into smaller parts.",
    sessionGrantMissing: "The secure cloud session did not reserve all required chunks.",
    decodeFailed: "The selected audio file could not be decoded.",
    clipboardUnavailable: "Clipboard access is unavailable in this browser.",
    copyFailed: "The transcript could not be copied to the clipboard.",
    exportFailed: "The document could not be created. Please try again.",
  },
};

const TR_COPY: AppCopy = {
  languageNames: {
    english: "İngilizce",
    turkish: "Türkçe",
    spanish: "İspanyolca",
    french: "Fransızca",
    german: "Almanca",
    italian: "İtalyanca",
    portuguese: "Portekizce",
    russian: "Rusça",
    arabic: "Arapça",
    hindi: "Hintçe",
    japanese: "Japonca",
    korean: "Korece",
  },
  freeBadge: "%100 Ücretsiz",
  noSignup: "Üyelik gerektirmez",
  appTitle: "Ses Transkripsiyonu",
  appSubtitle: "Ses dosyalarını anında metne çevirin. MP3, WAV, M4A ve daha fazlasını destekler.",
  stepLanguage: "Ses dili",
  selectLanguage: "Ses dilini seçin",
  stepUpload: "Ses dosyası",
  processing: "İşleniyor...",
  cancel: "İptal",
  cancelling: "İptal ediliyor...",
  compilingEngine: "GPU gölgelendiricileri derleniyor - yalnızca ilk kullanımda...",
  downloadingEngine: (downloaded, total) => `Yapay zeka motoru indiriliyor - ${downloaded} / ${total} MB`,
  preparingEnginePercent: (percent) => `Yapay zeka motoru hazırlanıyor - %${percent}`,
  preparingEngine: "Yapay zeka motoru hazırlanıyor...",
  transcriptionInProgress: "Transkripsiyon sürüyor",
  warmupDescription: "İlk bölüm işleniyor - yerel motor hazırlanıyor. İlk kullanımda 30-90 saniye sürebilir.",
  elapsed: (seconds) => `${seconds} sn geçti`,
  slice: (current, total) => `Dilim ${current} / ${total}`,
  segmentCount: (processed, total) => `${processed} / ${total} bölüm`,
  audioMinutes: (minutes) => `~${minutes} dk ses`,
  keepTabActive: "İşlem sürerken bu sekmeyi açık tutun.",
  decodingAudio: "Ses çözümleniyor...",
  downloadingModel: (downloaded, total, percent) => `Model indiriliyor... ${downloaded} / ${total} MB (%${percent})`,
  downloadingModelPercent: (percent) => `Model indiriliyor... %${percent}`,
  minuteValue: (minutes, seconds) => seconds === 0 ? `${minutes} dk` : `${minutes}:${String(seconds).padStart(2, "0")} dk`,
  transcribedRange: (processed, total) => `${processed} / ${total} yazıya çevrildi`,
  roughEstimate: (minutes, low, high) => `Ses uzunluğu ~${minutes} dk - tahmini işlem süresi: ${low}-${high} dk`,
  etaCalculating: "Tahmini süre hesaplanıyor...",
  etaFinishing: "İşlem tamamlanıyor...",
  etaRemaining: (timestamp) => `Tahmini kalan süre: ${timestamp}`,
  uploadToCloud: "Buluta yükleniyor...",
  secureSessionRetry: (seconds) => `Güvenli oturum kurulamadı. ${seconds} sn sonra yeniden denenecek...`,
  secureSessionBusy: (seconds) => `Güvenli oturum meşgul. ${seconds} sn sonra yeniden denenecek...`,
  uploadRetry: (seconds) => `Yükleme durakladı. ${seconds} sn sonra yeniden denenecek...`,
  cloudConnectionRetry: (seconds) => `Bulut bağlantısı kesildi. ${seconds} sn sonra yeniden denenecek...`,
  cloudRateLimitRetry: (seconds) => `Bulut kullanım sınırına ulaşıldı. ${seconds} sn sonra yeniden denenecek...`,
  cloudBusyRetry: (seconds) => `Bulut hizmeti meşgul. ${seconds} sn sonra yeniden denenecek...`,
  transcribingChunk: (current, total) => `${current} / ${total}. parça yazıya çevriliyor...`,
  transcribingAudio: "Ses yazıya çevriliyor...",
  longRecording: "Uzun kayıt algılandı. Mobil uyumlu parçalara ayrılıyor...",
  preparingUpload: "Ses yükleme için hazırlanıyor...",
  startingSecureSession: "Güvenli bulut oturumu başlatılıyor...",
  preparingChunk: (current, total) => `${current} / ${total}. parça hazırlanıyor...`,
  preparingAudio: "Ses hazırlanıyor...",
  uploadingChunk: (current, total) => `${current} / ${total}. parça güvenli biçimde yükleniyor...`,
  uploadingSecurely: "Ses güvenli biçimde yükleniyor...",
  transcribingPlaceholder: "Sesiniz yazıya çevriliyor...",
  textWillAppear: "Metin kısa süre içinde burada görünmeye başlayacak",
  rotatingPhrases: [
    "Metne çevirmek için bir ses dosyası yükleyin",
    "MP3, WAV, M4A ve 6 farklı format desteklenir",
    "Masaüstünde transkripsiyon tamamen yerel çalışır",
    "Whisper ve Moonshine yapay zeka modellerini kullanır",
    "Altyazılar için zaman damgalı çıktı alın",
    "Hesap gerekmez - tamamen ücretsizdir",
    "Dersler, toplantılar ve podcast'ler için kullanın",
    "Metin, zaman damgası, JSON, Word veya PDF olarak dışa aktarın",
    "12 farklı dilde transkripsiyon yapın",
    "Masaüstünde sesiniz cihazınızdan ayrılmaz",
  ],
  transcriptWillAppear: "Transkript burada görünecek",
  transcriptPanelTitle: "Transkript",
  transcriptReady: "Ses dosyası bekleniyor",
  wordCount: (count) => `${count} kelime`,
  plainText: "Düz metin",
  withTimestamps: "Zaman damgalı",
  copied: "Kopyalandı",
  copyText: "Metni kopyala",
  timestamps: "Zaman damgaları",
  exportJson: "JSON",
  continueWithAi: "Yapay zeka ile devam et",
  continueWithAiDescription: "Transkripti bir yapay zeka sohbetinde açın. Metin otomatik olarak panoya kopyalanır.",
  opensInNewTab: (service) => `Transkripti ${service} içinde aç (yeni sekmede açılır)`,
  developedBy: "Geliştiren",
  privacyPolicy: "Gizlilik Politikası",
  documentTitle: "Doküman başlığı",
  documentTitleDescription: "Dokümanınıza başlık ekleyin veya başlıksız indirmek için boş bırakın.",
  documentTitlePlaceholder: "Başlık girin (isteğe bağlı)",
  documentTitleInputLabel: "Doküman başlığı (isteğe bağlı)",
  downloadWord: "Word indir",
  downloadPdf: "PDF indir",
  transcriptLabel: "Transkripsiyon çıktısı",
  transcriptViewLabel: "Transkript görünüm biçimi",
  copyTextSuccess: "Metin kopyalandı",
  copyTimestampSuccess: "Zaman damgalı metin kopyalandı",
  jsonDownloaded: "JSON indirildi",
  transcriptionComplete: "Transkripsiyon tamamlandı",
  uploadDisabledLanguage: "Dosya yüklemeden önce ses dilini seçin.",
  uploadDisabledModel: "Yerel transkripsiyon motorunun yüklenmesini bekleyin.",
  dropzone: {
    unsupportedType: "Desteklenmeyen dosya türü. MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM veya OPUS yükleyin.",
    uploadLabel: "Ses dosyası yükle",
    dropHere: "Ses dosyanızı buraya bırakın",
    uploadPrompt: "Yazıya çevirmek için ses dosyası yükleyin (MP3, WAV, M4A)",
    instantConversion: "Sesi anında metne çevirin",
    or: "veya",
    browse: "cihazınızdan seçin",
    acceptedFormats: "Kabul edilen formatlar: .mp3, .wav, .m4a, .mp4, .ogg, .flac, .aac, .webm, .opus",
    selectedFile: (name) => `Seçilen dosya: ${name}`,
  },
  errors: {
    webAudioUnsupported: "Tarayıcınız Web Audio API özelliğini desteklemiyor.",
    workersUnsupported: "Tarayıcınız Web Worker özelliğini desteklemiyor.",
    workerUnexpected: "Transkripsiyon motorunda beklenmeyen bir hata oluştu.",
    workerUnavailable: "Transkripsiyon motoru kullanılamıyor.",
    sessionStartFailed: "Güvenli transkripsiyon oturumu başlatılamadı.",
    cloudUploadFailed: "Birden fazla denemeden sonra bulut yüklemesi başarısız oldu.",
    cloudTranscriptionFailed: "Bulut transkripsiyonu başarısız oldu. Lütfen yeniden deneyin.",
    cloudRequestFailed: (status) => `Bulut transkripsiyonu başarısız oldu (HTTP ${status}).`,
    mobileTooLong: "Bu kayıt mobil kullanım için 2 saatten uzun. Parçalara ayırın veya masaüstü kullanın.",
    mobileTooLarge: "Bu dosya mobil tarayıcıda işlenemeyecek kadar büyük. Masaüstü kullanın veya kaydı sıkıştırın.",
    localTooLong: "Bu kayıt 2 saatten uzun. Daha küçük parçalara ayırın.",
    localTooLarge: "Bu dosya tarayıcıda güvenli biçimde çözümlenemeyecek kadar büyük. Sıkıştırın veya parçalara ayırın.",
    sessionGrantMissing: "Güvenli bulut oturumu gerekli tüm parçaları ayıramadı.",
    decodeFailed: "Seçilen ses dosyası çözümlenemedi.",
    clipboardUnavailable: "Tarayıcınızda pano erişimi kullanılamıyor.",
    copyFailed: "Transkript panoya kopyalanamadı.",
    exportFailed: "Doküman oluşturulamadı. Lütfen yeniden deneyin.",
  },
};

export const APP_COPY: Record<AppLocale, AppCopy> = {
  en: EN_COPY,
  tr: TR_COPY,
};
