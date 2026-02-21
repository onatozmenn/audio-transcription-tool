const emptyModuleAlias = "./lib/empty-module.ts";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
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

export default nextConfig;
