import type { Metadata } from "next";
import Link from "next/link";
import { ShieldCheck } from "lucide-react";

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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="font-display text-base font-semibold text-neutral-100">{title}</h2>
      <div className="space-y-2 text-sm leading-7 text-neutral-400">{children}</div>
    </section>
  );
}

export default function TurkishPrivacyPolicy() {
  return (
    <main className="flex min-h-screen items-start justify-center px-4 py-12 sm:px-6">
      <article className="w-full max-w-2xl rounded-lg border border-white/[0.09] bg-[#171a18]/95 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.35)] sm:p-10">
        <Link href="/tr" className="mb-8 inline-flex items-center gap-1.5 text-xs text-neutral-500 transition-colors hover:text-neutral-300">
          <svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" height="14" viewBox="0 -960 960 960" width="14" fill="currentColor">
            <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
          </svg>
          Uygulamaya dön
        </Link>

        <header className="mb-8 space-y-3">
          <p className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
            <ShieldCheck className="size-3.5" />
            Gizlilik odaklı tasarım
          </p>
          <h1 className="font-display text-2xl font-semibold text-white">Gizlilik Politikası</h1>
          <p className="text-xs text-neutral-500">Son güncelleme: 14 Temmuz 2026</p>
        </header>

        <div className="space-y-8 divide-y divide-white/5">
          <div className="pt-8">
            <Section title="Genel Bakış">
              <p>
                Bu araç hesap gerektirmez; reklam veya analiz takipçisi kullanmaz. Yalnızca transkripsiyon hizmetini sunmak ve kötüye kullanımı önlemek için sınırlı veri işler.
              </p>
              <ul className="list-disc space-y-2 pl-5">
                <li><span className="font-medium text-neutral-200">Masaüstünde:</span> Ses, Whisper Small veya Moonshine ile tamamen tarayıcınızda işlenir. Ses ve transkript sunucuya gönderilmez.</li>
                <li><span className="font-medium text-neutral-200">Mobilde:</span> Ses geçici Vercel Blob depolamasına yüklenir, Groq Cloud üzerindeki Whisper Large V3 ile işlenir ve ardından silinir.</li>
              </ul>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Transkripsiyon ve Silme Süreci">
              <p><span className="font-medium text-neutral-200">Masaüstü:</span> Model ilk kullanımda Hugging Face üzerinden indirilip tarayıcı önbelleğine alınır. Sonraki işlemler internet bağlantısı olmadan çalışabilir.</p>
              <p><span className="font-medium text-neutral-200">Mobil:</span> Ses mobil uyumlu parçalara ayrılabilir. Groq yalnızca süreli bir bağlantıyla geçici dosyaya erişir. Başarılı işlemlerde dosya hemen silinir. Yarım kalan veya başarısız yüklemeler, dört saatlik oturum süresi ve bir saatlik ek sürenin ardından günlük temizlik göreviyle kaldırılır.</p>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Kötüye Kullanım Önleme ve İstek Verileri">
              <p>
                Mobil API istekleri Vercel BotID ile kontrol edilir ve hız sınırına tabidir. Uygulama, istek IP adresini paylaşımlı hız sınırı deposuna yazmadan önce anahtarlı HMAC değerine dönüştürür; ham IP adresi uygulama tarafından saklanmaz. Sayaçlar en fazla 30 dakika içinde silinir.
              </p>
              <p>Barındırma ve API sağlayıcıları standart ağ günlüklerini kendi gizlilik ve saklama politikalarına göre işleyebilir.</p>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Dışa Aktarma ve Yapay Zeka ile Devam">
              <p>Metin, zaman damgası, JSON, Word ve PDF çıktıları tarayıcınızda oluşturulur; dışa aktarma için sunucuya veri gönderilmez.</p>
              <p>ChatGPT, Claude, Gemini veya Grok düğmesine bastığınızda transkript panoya kopyalanır ve seçilen üçüncü taraf yeni sekmede açılır. Metin bu servislere otomatik gönderilmez; ilgili servisin kendi gizlilik politikası geçerlidir.</p>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Tarayıcı Önbelleği ve Yerel Depolama">
              <p>Yapay zeka model dosyaları tarayıcı önbelleğinde tutulabilir. Tarayıcı ayarlarınızdan bu verileri silebilirsiniz.</p>
              <p><code className="rounded bg-neutral-800 px-1 py-0.5 text-xs font-mono text-neutral-300">localStorage</code> içinde yalnızca modelin daha önce yüklenip yüklenmediğini belirten bir bayrak tutulur. Ses, transkript veya takip kimliği burada saklanmaz.</p>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Üçüncü Taraf Hizmetler">
              <ul className="list-disc space-y-2 pl-5">
                <li><span className="font-medium text-neutral-200">Hugging Face:</span> Masaüstü model dosyalarının ilk indirmesi.</li>
                <li><span className="font-medium text-neutral-200">Vercel Blob ve BotID:</span> Mobil geçici depolama ve kötüye kullanım önleme.</li>
                <li><span className="font-medium text-neutral-200">Upstash Redis:</span> En fazla 30 dakika yaşayan, anahtarlı anonim hız sınırı sayaçları. Ses veya transkript içermez.</li>
                <li><span className="font-medium text-neutral-200">Groq Cloud:</span> Mobil sesi Whisper Large V3 ile işler. İşlem sırasında <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">Groq Gizlilik Politikası</a> geçerlidir.</li>
              </ul>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="Çocukların Gizliliği">
              <p>Hizmet çocuklara yönelik değildir ve çocuklar hakkında bilerek profil oluşturmaz. Yukarıda açıklanan sınırlı işlemler tüm istekler için aynıdır.</p>
            </Section>
          </div>

          <div className="pt-8">
            <Section title="İletişim ve Değişiklikler">
              <p>Politika değişirse yeni tarih bu sayfada yayımlanır. Sorularınız için <a href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">LinkedIn</a> üzerinden iletişime geçebilirsiniz.</p>
            </Section>
          </div>
        </div>

        <footer className="mt-10 flex items-center justify-between border-t border-white/10 pt-6">
          <p className="text-xs text-neutral-500">Geliştiren <span className="font-medium text-neutral-300">Onat Özmen</span></p>
          <Link href="/tr" className="inline-flex items-center rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 hover:text-cyan-200">
            Uygulamaya dön
          </Link>
        </footer>
      </article>
    </main>
  );
}
