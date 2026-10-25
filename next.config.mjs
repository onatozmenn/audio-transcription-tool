import { withBotId } from "botid/next/config";

const emptyModuleAlias = "./lib/empty-module.ts";

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Content-Security-Policy", value: "frame-ancestors 'none'" },
  {
    key: "Permissions-Policy",
    value: "camera=(), geolocation=(), payment=(), usb=(), microphone=(self)",
  },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  serverExternalPackages: ["@huggingface/transformers"],
  turbopack: {
    resolveAlias: {
      sharp: emptyModuleAlias,
      "onnxruntime-node": emptyModuleAlias,
    },
  },
  webpack: (config, { isServer }) => {
    config.experiments = {
      ...(config.experiments ?? {}),
      asyncWebAssembly: true,
      layers: true,
    };

    config.resolve.alias = {
      ...(config.resolve.alias ?? {}),
      "sharp$": false,
      "onnxruntime-node$": false,
    };

    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
    });

    config.output = {
      ...(config.output ?? {}),
      webassemblyModuleFilename: isServer
        ? "../static/wasm/[modulehash].wasm"
        : "static/wasm/[modulehash].wasm",
    };

    return config;
  },
};

export default withBotId(nextConfig);
