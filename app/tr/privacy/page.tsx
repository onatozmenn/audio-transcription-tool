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
    canonical: "https://www.audio-transcription.app/tr/privacy",
    languages: {
      en: "https://www.audio-transcription.app/privacy",
      tr: "https://www.audio-transcription.app/tr/privacy",
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
    steps: ["Cihaz", "Geçici yükleme", "Groq", "Silme"],
    note: "Başarılı dosyalar hemen; yarım kalan dosyalar zamanlanmış temizlikle silinir.",
  },
  {
    label: "Hizmet koruması",
    steps: ["BotID", "HMAC sayaç", "Sona erme"],
    note: "Ham IP saklanmaz. Hız sınırı ve günlük kullanım sayaçları en fazla 24 saat yaşar.",
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
            <strong>Mobilde:</strong> Ses geçici bulut işlemeye gönderilir (Vercel Blob depolaması veya iPhone Safari&apos;de doğrudan sunucumuz üzerinden), Groq Cloud üzerindeki Whisper Large V3 ile işlenir ve ardından silinir.
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
          <strong>Mobil:</strong> Ses cihazınızda parçalara ayrılır ve sıkıştırılır. Çoğu tarayıcıda her parça rastgele ve tahmin edilemeyen bir adresle geçici Vercel Blob depolamasına yüklenir; Groq dosyayı bu adresten alır. iPhone Safari&apos;de parçalar saklanmadan sunucumuz üzerinden doğrudan Groq&apos;a iletilir. Başarılı parçalar hemen silinir. Yarım kalan veya başarısız yüklemeler, beş saatten eski olduklarında günlük temizlik göreviyle kaldırılır; yani en fazla yaklaşık 29 saat kalır.
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
          Mobil API istekleri Vercel BotID ile kontrol edilir ve hız sınırına tabidir. Uygulama, istek IP adresini paylaşımlı hız sınırı deposuna yazmadan önce anahtarlı HMAC değerine dönüştürür; ham IP adresi uygulama tarafından saklanmaz. Kısa süreli hız sınırı sayaçları 30 dakika içinde, günlük bulut kullanım sayacı (ses içeriği olmadan, yalnızca işlenen ses miktarı) 24 saat içinde silinir.
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
          Yapay zeka model dosyaları (modele ve cihaza göre yaklaşık 250-410 MB) ilk indirmeden sonra tarayıcı önbelleğinde tutulur. Tarayıcı ayarlarınızdan bu verileri silebilirsiniz.
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
        <li><strong>Vercel (barındırma, Blob ve BotID):</strong> Barındırma, mobil geçici depolama ve kötüye kullanım önleme.</li>
        <li><strong>Upstash Redis:</strong> En fazla 24 saat yaşayan, anahtarlı takma adlı hız sınırı ve günlük kullanım sayaçları. Ses veya transkript içermez.</li>
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
        Politika değişirse yeni tarih bu sayfada yayımlanır. Sorularınız için <a href="https://www.linkedin.com/in/onat-ozmen" target="_blank" rel="noopener noreferrer">LinkedIn</a> üzerinden iletişime geçebilir veya <a href="https://github.com/onatozmenn/audio-transcription-tool/issues" target="_blank" rel="noopener noreferrer">GitHub</a>&apos;da issue açabilirsiniz. Kaynak kodu açıktır (MIT Lisansı); verinizin nasıl işlendiğini kendiniz doğrulayabilirsiniz.
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
      updated="25 Ekim 2026"
    />
  );
}
