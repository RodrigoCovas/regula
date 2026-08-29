/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        // The readiness endpoint sits outside /api on the backend (issue #45);
        // the frontend's readiness check proxies it the same way (issue #48).
        source: "/readiness",
        destination: `${backendUrl}/readiness`,
      },
    ];
  },
};

export default nextConfig;
