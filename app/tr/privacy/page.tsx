import type { Metadata } from "next";
import {
  PrivacyPolicyPage,
  type PrivacyFact,
  type PrivacySection,
} from "@/components/privacy-policy-page";

export const metadata: Metadata = {
  title: { absolute: "Gizlilik Politikası - Ses Transkripsiyonu" },
  description: "Yerel ve bulut transkripsiyonunda verilerin nasıl işlendiği, korunduğu ve silindiği.",
  alternates: {
    canonical: "https://audio-transcription.app/tr/privacy",
    languages: {
      en: "https://audio-transcription.app/privacy",
      tr: "https://audio-transcription.app/tr/privacy",
    },
  },
};

const facts: PrivacyFact[] = [
  {
    label: "Masaüstü",
    steps: ["Cihaz", "Tarayıcı modeli", "Transkript"],
    note: "Ses ve transkript sunucuya gönderilmez.",
  },
  {
    label: "Mobil",
    steps: ["Cihaz", "Geçici Blob", "Groq", "Silme"],
    note: "Başarılı dosyalar hemen; yarım kalan dosyalar zamanlanmış temizlikle silinir.",
  },
  {
    label: "Hizmet koruması",
    steps: ["BotID", "HMAC sayaç", "Sona erme"],
    note: "Ham IP saklanmaz. Hız sınırı sayaçları en fazla 30 dakika yaşar.",
  },
];

const sections: PrivacySection[] = [
  {
    id: "genel-bakis",
    title: "Genel Bakış",
    body: (
      <>
        <p>
          Bu araç hesap gerektirmez; reklam veya analiz takipçisi kullanmaz. Yalnızca transkripsiyon hizmetini sunmak ve kötüye kullanımı önlemek için sınırlı veri işler.
        </p>
        <ul>
          <li>
            <strong>Masaüstünde:</strong> Ses, Whisper Small veya Moonshine ile tamamen tarayıcınızda işlenir. Ses ve transkript sunucuya gönderilmez.
          </li>
          <li>
            <strong>Mobilde:</strong> Ses geçici Vercel Blob depolamasına yüklenir, Groq Cloud üzerindeki Whisper Large V3 ile işlenir ve ardından silinir.
          </li>
        </ul>
      </>
    ),
  },
  {
    id: "transkripsiyon-ve-silme",
    title: "Transkripsiyon ve Silme Süreci",
    body: (
      <>
        <p>
          <strong>Masaüstü:</strong> Model ilk kullanımda Hugging Face üzerinden indirilip tarayıcı önbelleğine alınır. Sonraki işlemler internet bağlantısı olmadan çalışabilir.
        </p>
        <p>
          <strong>Mobil:</strong> Ses mobil uyumlu parçalara ayrılabilir. Groq yalnızca süreli bir bağlantıyla geçici dosyaya erişir. Başarılı işlemlerde dosya hemen silinir. Yarım kalan veya başarısız yüklemeler, dört saatlik oturum süresi ve bir saatlik ek sürenin ardından günlük temizlik göreviyle kaldırılır.
        </p>
      </>
    ),
  },
  {
    id: "istek-verileri",
    title: "Kötüye Kullanım Önleme ve İstek Verileri",
    body: (
      <>
        <p>
          Mobil API istekleri Vercel BotID ile kontrol edilir ve hız sınırına tabidir. Uygulama, istek IP adresini paylaşımlı hız sınırı deposuna yazmadan önce anahtarlı HMAC değerine dönüştürür; ham IP adresi uygulama tarafından saklanmaz. Sayaçlar en fazla 30 dakika içinde silinir.
        </p>
        <p>
          Barındırma ve API sağlayıcıları standart ağ günlüklerini kendi gizlilik ve saklama politikalarına göre işleyebilir.
        </p>
      </>
    ),
  },
  {
    id: "disa-aktarma",
    title: "Dışa Aktarma ve Yapay Zeka ile Devam",
    body: (
      <>
        <p>
          Metin, zaman damgası, JSON, Word ve PDF çıktıları tarayıcınızda oluşturulur; dışa aktarma için sunucuya veri gönderilmez.
        </p>
        <p>
          ChatGPT, Claude, Gemini veya Grok düğmesine bastığınızda transkript panoya kopyalanır ve seçilen üçüncü taraf yeni sekmede açılır. Metin bu servislere otomatik gönderilmez; ilgili servisin kendi gizlilik politikası geçerlidir.
        </p>
      </>
    ),
  },
  {
    id: "tarayici-depolama",
    title: "Tarayıcı Önbelleği ve Yerel Depolama",
    body: (
      <>
        <p>
          Yapay zeka model dosyaları tarayıcı önbelleğinde tutulabilir. Tarayıcı ayarlarınızdan bu verileri silebilirsiniz.
        </p>
        <p>
          <code>localStorage</code> içinde yalnızca modelin daha önce yüklenip yüklenmediğini belirten bir bayrak tutulur. Ses, transkript veya takip kimliği burada saklanmaz.
        </p>
      </>
    ),
  },
  {
    id: "ucuncu-taraflar",
    title: "Üçüncü Taraf Hizmetler",
    body: (
      <ul>
        <li><strong>Hugging Face:</strong> Masaüstü model dosyalarının ilk indirmesi.</li>
        <li><strong>Vercel Blob ve BotID:</strong> Mobil geçici depolama ve kötüye kullanım önleme.</li>
        <li><strong>Upstash Redis:</strong> En fazla 30 dakika yaşayan, anahtarlı anonim hız sınırı sayaçları. Ses veya transkript içermez.</li>
        <li>
          <strong>Groq Cloud:</strong> Mobil sesi Whisper Large V3 ile işler. İşlem sırasında <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer">Groq Gizlilik Politikası</a> geçerlidir.
        </li>
      </ul>
    ),
  },
  {
    id: "cocuklarin-gizliligi",
    title: "Çocukların Gizliliği",
    body: (
      <p>
        Hizmet çocuklara yönelik değildir ve çocuklar hakkında bilerek profil oluşturmaz. Yukarıda açıklanan sınırlı işlemler tüm istekler için aynıdır.
      </p>
    ),
  },
  {
    id: "iletisim-ve-degisiklikler",
    title: "İletişim ve Değişiklikler",
    body: (
      <p>
        Politika değişirse yeni tarih bu sayfada yayımlanır. Sorularınız için <a href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250" target="_blank" rel="noopener noreferrer">LinkedIn</a> üzerinden iletişime geçebilirsiniz.
      </p>
    ),
  },
];

export default function TurkishPrivacyPolicy() {
  return (
    <PrivacyPolicyPage
      activeLanguage="TR"
      alternateHref="/privacy"
      alternateLanguage="EN"
      backHref="/tr"
      backLabel="Uygulamaya dön"
      contactLabel="LinkedIn"
      contentsLabel="Bu sayfada"
      dataPathTitle="Verinizin izlediği yol"
      developedBy="Geliştiren"
      facts={facts}
      languageLabel="Politika dili"
      lastUpdatedLabel="Son güncelleme"
      scopeLabel="Kapsam"
      scopeText="Masaüstünde yerel, mobilde geçici bulut işleme"
      sections={sections}
      summary="Sesinizin ne zaman cihazınızda kaldığını, mobil işlem sırasında nereye gittiğini ve geçici dosyaların ne zaman silindiğini açıkça anlatır."
      title="Gizlilik Politikası"
      updated="14 Temmuz 2026"
    />
  );
}
