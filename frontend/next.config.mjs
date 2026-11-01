/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output keeps the production Docker image small.
  output: "standalone",
  reactStrictMode: true,
  devIndicators: false,
};

export default nextConfig;
