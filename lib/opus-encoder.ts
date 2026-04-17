/**
 * Offline WebCodecs Opus encoder with a minimal Ogg container muxer.
 *
 * Used on the mobile cloud path to replace uncompressed 16-bit PCM WAV
 * chunks (~115 MB / hour) with Opus (~10 MB / hour at 24 kbps), cutting
 * Blob egress and Groq fetch bandwidth by ~10×.
 *
 * Browser support: Chrome 94+, Edge 94+, Safari 16.4+, Android Chrome.
 * Falls back to WAV on older browsers via `isOpusOggEncodingSupported()`.
 */

// ─── Ogg CRC32 (poly 0x04c11db7, MSB-first, no reflection) ────────────────────
const OGG_CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let r = i << 24;
    for (let j = 0; j < 8; j += 1) {
      r = (r & 0x80000000) !== 0 ? (r << 1) ^ 0x04c11db7 : r << 1;
    }
    table[i] = r >>> 0;
  }
  return table;
})();

function oggCRC(data: Uint8Array): number {
  let crc = 0;
  for (let i = 0; i < data.length; i += 1) {
    crc = (OGG_CRC_TABLE[((crc >>> 24) ^ data[i]) & 0xff] ^ (crc << 8)) >>> 0;
  }
  return crc;
}

function writeOggPage(
  payload: Uint8Array,
  segmentTable: Uint8Array,
  headerType: number,
  granulePosition: bigint,
  serialNumber: number,
  pageSequence: number,
): Uint8Array {
  const page = new Uint8Array(27 + segmentTable.length + payload.length);
  const dv = new DataView(page.buffer);

  // "OggS" capture pattern
  page[0] = 0x4f;
  page[1] = 0x67;
  page[2] = 0x67;
  page[3] = 0x53;
  page[4] = 0; // stream structure version
  page[5] = headerType; // 0x02 BOS, 0x00 cont, 0x04 EOS
  dv.setBigInt64(6, granulePosition, true);
  dv.setUint32(14, serialNumber, true);
  dv.setUint32(18, pageSequence, true);
  dv.setUint32(22, 0, true); // checksum placeholder
  page[26] = segmentTable.length;
  page.set(segmentTable, 27);
  page.set(payload, 27 + segmentTable.length);

  dv.setUint32(22, oggCRC(page), true);
  return page;
}

function lacingFor(length: number): number[] {
  const out: number[] = [];
  let remaining = length;
  while (remaining >= 255) {
    out.push(255);
    remaining -= 255;
  }
  out.push(remaining);
  return out;
}

function buildOpusHead(inputSampleRate: number, channels: number, preSkip: number): Uint8Array {
  const buf = new Uint8Array(19);
  const dv = new DataView(buf.buffer);
  // "OpusHead"
  buf.set([0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64], 0);
  buf[8] = 1; // version
  buf[9] = channels;
  dv.setUint16(10, preSkip, true);
  dv.setUint32(12, inputSampleRate, true);
  dv.setInt16(16, 0, true); // output gain
  buf[18] = 0; // channel mapping family: mono/stereo
  return buf;
}

function buildOpusTags(): Uint8Array {
  const vendor = new TextEncoder().encode("audio-transcription-tool");
  const buf = new Uint8Array(8 + 4 + vendor.length + 4);
  const dv = new DataView(buf.buffer);
  // "OpusTags"
  buf.set([0x4f, 0x70, 0x75, 0x73, 0x54, 0x61, 0x67, 0x73], 0);
  dv.setUint32(8, vendor.length, true);
  buf.set(vendor, 12);
  dv.setUint32(12 + vendor.length, 0, true); // 0 user comments
  return buf;
}

// ─── Public API ───────────────────────────────────────────────────────────────

const OPUS_ALLOWED_SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000] as const;

export function isOpusOggEncodingSupported(): boolean {
  if (typeof globalThis === "undefined") return false;
  const g = globalThis as unknown as {
    AudioEncoder?: unknown;
    AudioData?: unknown;
  };
  return typeof g.AudioEncoder !== "undefined" && typeof g.AudioData !== "undefined";
}

/**
 * Encode a mono Float32 PCM buffer to an Ogg/Opus Blob.
 * Caller must ensure `sampleRate` is one of {8000,12000,16000,24000,48000}.
 */
export async function encodeFloat32ToOpusOgg(
  samples: Float32Array,
  sampleRate: number,
  bitrate = 24_000,
): Promise<Blob> {
  if (!isOpusOggEncodingSupported()) {
    throw new Error("WebCodecs Opus encoding is not supported in this browser.");
  }
  if (!OPUS_ALLOWED_SAMPLE_RATES.includes(sampleRate as (typeof OPUS_ALLOWED_SAMPLE_RATES)[number])) {
    throw new Error(`Unsupported Opus sample rate: ${sampleRate}`);
  }

  const packets: Uint8Array[] = [];
  const packetGranules48k: number[] = [];
  let encoderError: Error | null = null;

  const EncoderCtor = (globalThis as unknown as { AudioEncoder: typeof AudioEncoder }).AudioEncoder;
  const AudioDataCtor = (globalThis as unknown as { AudioData: typeof AudioData }).AudioData;

  const encoder = new EncoderCtor({
    output: (chunk: EncodedAudioChunk) => {
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      packets.push(data);
      // Opus decoder always runs at 48 kHz regardless of input rate.
      // Granule position advances in 48 kHz samples.
      const durationSec = (chunk.duration ?? 20_000) / 1_000_000;
      packetGranules48k.push(Math.round(durationSec * 48_000));
    },
    error: (e: Error) => {
      encoderError = e;
    },
  });

  encoder.configure({
    codec: "opus",
    sampleRate,
    numberOfChannels: 1,
    bitrate,
  });

  // Copy to a detached buffer so AudioData's internal close() won't touch
  // the caller's Float32Array.
  const buffer = new Float32Array(samples.length);
  buffer.set(samples);

  const audioData = new AudioDataCtor({
    format: "f32",
    sampleRate,
    numberOfFrames: buffer.length,
    numberOfChannels: 1,
    timestamp: 0,
    data: buffer,
  });

  encoder.encode(audioData);
  audioData.close();

  await encoder.flush();
  encoder.close();

  if (encoderError) throw encoderError;
  if (packets.length === 0) {
    throw new Error("Opus encoder produced no packets.");
  }

  // ── Mux into Ogg ────────────────────────────────────────────────────────────
  const serialNumber = Math.floor(Math.random() * 0xffffffff) >>> 0;
  // Opus reference pre-skip (80 ms @ 48 kHz).
  const preSkip = 3840;

  const pages: Uint8Array[] = [];
  let pageSeq = 0;

  const opusHead = buildOpusHead(sampleRate, 1, preSkip);
  pages.push(
    writeOggPage(
      opusHead,
      new Uint8Array(lacingFor(opusHead.length)),
      0x02,
      BigInt(0),
      serialNumber,
      pageSeq++,
    ),
  );

  const opusTags = buildOpusTags();
  pages.push(
    writeOggPage(
      opusTags,
      new Uint8Array(lacingFor(opusTags.length)),
      0x00,
      BigInt(0),
      serialNumber,
      pageSeq++,
    ),
  );

  let cumulativeGranule = BigInt(0);
  let idx = 0;
  while (idx < packets.length) {
    const pagePackets: Uint8Array[] = [];
    const pageSegTable: number[] = [];

    while (idx < packets.length) {
      const pkt = packets[idx];
      const lacingBytes = lacingFor(pkt.length);
      // Each Ogg page holds at most 255 lacing values.
      if (pageSegTable.length + lacingBytes.length > 255) break;

      pageSegTable.push(...lacingBytes);
      pagePackets.push(pkt);
      cumulativeGranule += BigInt(packetGranules48k[idx]);
      idx += 1;
    }

    const payloadSize = pagePackets.reduce((acc, p) => acc + p.length, 0);
    const payload = new Uint8Array(payloadSize);
    let offset = 0;
    for (const p of pagePackets) {
      payload.set(p, offset);
      offset += p.length;
    }

    const isLast = idx >= packets.length;
    pages.push(
      writeOggPage(
        payload,
        new Uint8Array(pageSegTable),
        isLast ? 0x04 : 0x00,
        cumulativeGranule,
        serialNumber,
        pageSeq++,
      ),
    );
  }

  const totalSize = pages.reduce((acc, p) => acc + p.length, 0);
  const out = new Uint8Array(totalSize);
  let offset = 0;
  for (const p of pages) {
    out.set(p, offset);
    offset += p.length;
  }

  return new Blob([out], { type: "audio/ogg" });
}
