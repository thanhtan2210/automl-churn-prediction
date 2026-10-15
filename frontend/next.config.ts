import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root to this folder (a stray lockfile in a parent folder would be picked otherwise).
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
